"""The cookbook's renderer for DeepSeek-V4.1-Flash and DeepSeek-V4.1-Flash-1017.

It follows each checkpoint's ``chat_template.jinja``: DeepSeek-V4.1-Flash at revision
314113f8a7dadb388215396ba5fbee2c610a04d6, and DeepSeek-V4.1-Flash-1017 at revision
08c8657f02291e11b6f6b63e0e16014f8e748518. Prime 0.2.0 has a renderer for V4 only. V4.1
renamed the tool-call tags, opens the prompt with ``<｜System｜>``, writes the reasoning
effort as a number and takes images. So this module writes the template's text itself,
and it uses Prime only for the result types, the turn-close helper and the scanner.

It renders what agents send, and it refuses the rest instead of copying every template
quirk. It refuses a system message after the first one, as Prime's V4.1 work does. It
takes images in user and tool messages, as the Qwen renderers do.

The template writes one ``<｜deepseek_image｜>`` per image, and the inference engine
expands it. The trainer needs the expanded form, so this renderer writes the token once
for each image token. The checkpoints ship no image processor. The count comes from the
image's size, as the trainer's ``deepseek_v41_image_tokens`` computes it.

The two checkpoints differ only in when a turn starts, which decides whose reasoning the
template drops. V4.1 starts a turn at every user message, tool results included. 1017
starts one only at a user message that holds text or an image. Neither drops reasoning
when the conversation declares tools.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from renderers.base import (
    MultiModalData,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    RenderedTokens,
    ToolCallParseStatus,
    extract_message_tool_names,
    trim_to_turn_close,
)
from renderers.tagged import DEEPSEEK_V4_GRAMMAR, TaggedGrammar, scan_tagged_response

Variant = Literal["v4.1", "1017"]

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
SYSTEM, USER, ASSISTANT = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>"
THINK_START, THINK_END = "<think>", "</think>"
IMAGE = "<｜deepseek_image｜>"
_DSML = "｜DSML｜"
_CALLS, _INVOKE, _PARAMETER = f"{_DSML} calls", f"{_DSML} invoke", f"{_DSML} parameter"

_TOOLS_HEADER = (
    "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
    f'You can invoke tools by writing a "<{_CALLS}>" block like the following:\n\n'
    f'<{_CALLS}>\n<{_INVOKE} name="$TOOL_NAME">\n'
    f'<{_PARAMETER} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{_PARAMETER}>\n'
    f'...\n</{_INVOKE}>\n<{_INVOKE} name="$TOOL_NAME2">\n...\n</{_INVOKE}>\n</{_CALLS}>\n\n'
    'String parameters should be specified as is and set `string="true"`. For all other '
    "types (numbers, booleans, arrays, objects), pass the value in JSON format and set "
    '`string="false"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST '
    "output your complete reasoning inside <think>...</think> BEFORE any tool calls or "
    "final response.\n\nOtherwise, output directly after </think> with tool calls or "
    "final response.\n\n### Available Tool Schemas\n\n"
)
_TOOLS_FOOTER = (
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to "
    "invoke tool calls.\n"
)

# vLLM's deepseek_v41 tool parser: V4's states, with V4.1's tag names.
_GRAMMAR = TaggedGrammar(
    DEEPSEEK_V4_GRAMMAR.transitions,
    {
        f"<{_CALLS}>": "section_start",
        f"</{_CALLS}>": "section_end",
        f"<{_INVOKE}": "tool_start",
        f"</{_INVOKE}>": "tool_end",
    },
)
_HEADER = re.compile(rf'<{re.escape(_INVOKE)} name="(.*?)">\n', re.DOTALL)
_PARAMETER_BLOCK = re.compile(
    rf'<{re.escape(_PARAMETER)} name="(.*?)" string="(true|false)">(.*?)</{re.escape(_PARAMETER)}>',
    re.DOTALL,
)

# The vision_config of both checkpoints. tests/contracts checks it against config.json.
PATCH_SIZE, DOWNSAMPLE_RATIO, MAX_IMAGE_TOKENS, MIN_PIXELS = 14, 3, 1024, 295936


def image_tokens(width: int, height: int) -> int:
    """Return the tokens one image spans, as the trainer's ``deepseek_v41_image_tokens`` does.

    The image is scaled up to at least ``MIN_PIXELS`` and padded to whole patches. Each
    row of merged patches ends in a newline slot, and a start and an end slot wrap the
    grid. An image over ``MAX_IMAGE_TOKENS`` shrinks, keeping its aspect, until it fits.
    """
    w: float = width
    h: float = height
    if 0 < w * h < MIN_PIXELS:
        scale = (MIN_PIXELS / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
    best_h, best_w = math.ceil(h / PATCH_SIZE) * PATCH_SIZE, math.ceil(w / PATCH_SIZE) * PATCH_SIZE
    if _span(best_h, best_w) > MAX_IMAGE_TOKENS:
        best_h, best_w = _shrink_to_budget(h, w)
    return _span(best_h, best_w)


def _span(height: float, width: float) -> int:
    rows = math.ceil((height // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    cols = math.ceil((width // PATCH_SIZE) / DOWNSAMPLE_RATIO)
    return rows * (cols + 1) + 2


def _shrink_to_budget(height: float, width: float) -> tuple[int, int]:
    cell = PATCH_SIZE * DOWNSAMPLE_RATIO
    aspect = height / width
    max_cols = math.sqrt((MAX_IMAGE_TOKENS - 2) / aspect + 0.25) - 0.5
    max_rows = max_cols * aspect
    if max_cols < 1.0:
        return (MAX_IMAGE_TOKENS - 2) // 2 * cell, cell
    if max_rows < 1.0:
        return cell, (MAX_IMAGE_TOKENS - 3) * cell
    scale = min(math.floor(max_cols) * cell / width, math.floor(max_rows) * cell / height)
    return (
        math.floor(height * scale / PATCH_SIZE) * PATCH_SIZE,
        math.floor(width * scale / PATCH_SIZE) * PATCH_SIZE,
    )


_BUDGETS = {"low": 50, "high": 75, "max": 100}


class DeepSeekV41RendererConfig(BaseModel):
    """The template options that ``renderer_chat_template_kwargs`` may set.

    Both templates read these names, and both give the named efforts the same budgets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    enable_thinking: bool = True
    drop_thinking: bool = True
    reasoning_effort: Literal["low", "high", "max"] | Annotated[int, Field(ge=1, le=100)] = "high"

    @property
    def budget(self) -> int:
        effort = self.reasoning_effort
        return _BUDGETS[effort] if isinstance(effort, str) else effort


# --- messages as the template merges them ---------------------------------------------

Piece = str | Mapping[str, Any]
"""Text, or an image part, in the order the template writes them."""


@dataclass
class _Block:
    pieces: list[Piece]
    owner: int
    call_id: Any = None  # set on a tool result


@dataclass
class _Turn:
    role: str
    owner: int
    blocks: list[_Block]
    message: Mapping[str, Any]


def _text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("DeepSeek-V4.1 text must be a string")
    if IMAGE in text:
        raise ValueError(f"text contains {IMAGE}; pass images as image parts")
    return text


def _pieces(content: Any, *, images: bool) -> list[Piece]:
    """Return a message's text and images, joined by blank lines as the template joins them."""
    if isinstance(content, str):
        return [_text(content)]
    if not isinstance(content, list):
        raise TypeError("DeepSeek-V4.1 content must be a string or a list of parts")
    from loops_cookbook.llm.images import is_image_part

    pieces: list[Piece] = []
    for part in content:
        if pieces:
            pieces.append("\n\n")
        if isinstance(part, Mapping) and part.get("type") == "text":
            pieces.append(_text(part.get("text") or ""))
        elif images and isinstance(part, Mapping) and is_image_part(part):
            pieces.append(part)
        else:
            raise ValueError(
                "DeepSeek-V4.1 takes text parts, and images in user and tool messages only"
            )
    return pieces


_REFUSED_KEYS = ("task", "wo_eos", "response_format", "tools", "content_blocks", "thinking")


def _merge(messages: Sequence[Mapping[str, Any]]) -> list[_Turn]:
    """Merge tool results and consecutive user messages into user turns, as the template does."""
    turns: list[_Turn] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        refused = [key for key in _REFUSED_KEYS if message.get(key) is not None]
        if refused:
            raise ValueError(f"the DeepSeek-V4.1 renderer does not render {refused[0]!r}")
        if role in ("user", "tool"):
            content = message.get("content")
            if role == "tool":
                if content is None:
                    raise ValueError("a DeepSeek-V4.1 tool message needs content")
                pieces = ["<tool_result>", *_pieces(content, images=True), "</tool_result>"]
                block = _Block(pieces, index, message.get("tool_call_id", ""))
            else:
                block = _Block(_pieces(content or "", images=True), index)
            if turns and turns[-1].role == "user":
                turns[-1].blocks.append(block)
            else:
                turns.append(_Turn("user", index, [block], {}))
        elif role == "system" and not turns:
            turns.append(_Turn(role, index, [], message))
        elif role == "assistant":
            _text(message.get("reasoning_content") or "")  # checked even where it's dropped
            turns.append(_Turn(role, index, [], message))
        else:
            raise ValueError(
                "the DeepSeek-V4.1 renderer takes one leading system message, then user, "
                f"assistant and tool messages, not {role!r} at messages[{index}]"
            )
    return turns


def _call_ids(message: Mapping[str, Any]) -> list[Any]:
    return [
        call.get("id") or (call.get("function") or {}).get("id", "")
        for call in message.get("tool_calls") or ()
    ]


def _order(call_ids: Sequence[Any], call_id: Any) -> int:
    """Return the position of the last call with this ID, or 0, as the template sorts by."""
    position = 0
    for index, candidate in enumerate(call_ids):
        if candidate and candidate == call_id:
            position = index
    return position


def _sorted_results(items: list, call_ids: Sequence[Any], call_id) -> list:
    """Sort the tool results among ``items`` into call order, and leave the rest in place."""
    results = [item for item in items if call_id(item) is not None]
    if len(results) < 2:
        return items
    ordered = iter(sorted(results, key=lambda item: _order(call_ids, call_id(item))))
    return [next(ordered) if call_id(item) is not None else item for item in items]


def _tool_schemas(tools: Sequence[Mapping[str, Any]]) -> str:
    schemas = []
    for tool in tools:
        function = tool.get("function", tool)
        if tool.get("namespace") is not None or function.get("namespace") is not None:
            raise ValueError("the DeepSeek-V4.1 renderer does not render tool namespaces")
        schemas.append(json.dumps(dict(function), ensure_ascii=False))
    return _TOOLS_HEADER + "\n".join(schemas) + _TOOLS_FOOTER


def _tool_calls(calls: Sequence[Mapping[str, Any]]) -> str:
    invokes = []
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, Mapping):
            raise TypeError("DeepSeek-V4.1 tool call arguments must be an object")
        lines = [
            f'<{_PARAMETER} name="{key}" string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            f"</{_PARAMETER}>"
            for key, value in arguments.items()
        ]
        invokes.append(
            f'<{_INVOKE} name="{function.get("name")}">\n' + "\n".join(lines) + f"\n</{_INVOKE}>"
        )
    return f"\n\n<{_CALLS}>\n" + "\n".join(invokes) + f"\n</{_CALLS}>"


def _runs(ids: Sequence[int], token: int) -> list[PlaceholderRange]:
    runs, at = [], 0
    while at < len(ids):
        if ids[at] == token:
            start = at
            while at < len(ids) and ids[at] == token:
                at += 1
            runs.append(PlaceholderRange(offset=start, length=at - start))
        else:
            at += 1
    return runs


def _images(ids: Sequence[int], token: int, new: Sequence[Mapping[str, Any]]):
    """Report every image run, with the parts of the ``new`` images, which come last."""
    ranges = _runs(ids, token)
    if len(ranges) < len(new):
        raise ValueError(f"rendered {len(new)} images but found {len(ranges)} image runs")
    if not ranges:
        return None
    return MultiModalData(
        mm_hashes={"image": [f"deepseek-{r.offset}" for r in ranges]},
        mm_placeholders={"image": ranges},
        mm_items={"image": [{} for _ in range(len(ranges) - len(new))] + [dict(p) for p in new]},
    )


# --- the renderer ---------------------------------------------------------------------


class DeepSeekV41Renderer:
    """Renders DeepSeek-V4.1-Flash and -1017 conversations as each one's chat template does."""

    reads_calls_whole = True  # a block is read whole or reported, never in part
    is_multimodal = True  # renderers.base.is_multimodal honours this.

    def __init__(self, tokenizer, variant: Variant, config: DeepSeekV41RendererConfig):
        self._tokenizer = tokenizer
        self.variant: Variant = variant
        self.config = config
        self._eos = self._token_id(EOS)
        self._think_start = self._token_id(THINK_START)
        self._think_end = self._token_id(THINK_END)
        self._image = self._token_id(IMAGE)
        self._image_cache: dict[str, int] = {}

    def _token_id(self, text: str) -> int:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text} is not one token in this tokenizer")
        return ids[0]

    def get_stop_token_ids(self) -> list[int]:
        return [self._eos]

    def _starts_turn(self, turn: _Turn) -> bool:
        """Whether a merged user turn bounds the reasoning that the template drops."""
        if turn.role != "user":
            return False
        return self.variant == "v4.1" or any(block.call_id is None for block in turn.blocks)

    # --- rendering --------------------------------------------------------------------

    def render(self, messages, *, tools=None, add_generation_prompt=False) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")
        turns = _merge(messages)
        if tools and turns[0].role != "system":
            turns.insert(0, _Turn("system", -1, [], {"content": ""}))
        call_ids: list[Any] = []
        for turn in turns:
            if turn.role == "assistant" and turn.message.get("tool_calls"):
                call_ids = _call_ids(turn.message)
            turn.blocks = _sorted_results(turn.blocks, call_ids, lambda block: block.call_id)
        thinking = self.config.enable_thinking
        drop = self.config.drop_thinking and not tools
        last_turn = max((i for i, turn in enumerate(turns) if self._starts_turn(turn)), default=-1)

        writer = _Writer()
        writer.write(BOS, -1)
        if thinking or turns[0].role == "system":
            writer.write(SYSTEM, -1)
        if thinking:
            writer.write(
                f"Reasoning Effort: {self.config.budget} (range 1-100, the higher the value, "
                "the more thorough the reasoning)\n\n",
                -1,
            )
        for index, turn in enumerate(turns):
            if turn.role == "system":
                writer.pieces(_pieces(turn.message.get("content") or "", images=False), turn.owner)
                if tools:
                    writer.write("\n\n" + _tool_schemas(tools), turn.owner)
            elif turn.role == "user":
                self._user(writer, turn)
            else:
                message = turn.message
                if thinking and (not drop or index > last_turn):
                    reasoning = message.get("reasoning_content") or ""
                    writer.write(reasoning + THINK_END, turn.owner, sampled=True)
                content = _pieces(message.get("content") or "", images=False)
                writer.pieces(content, turn.owner, sampled=True)
                if message.get("tool_calls"):
                    writer.write(_tool_calls(message["tool_calls"]), turn.owner, sampled=True)
                writer.write(EOS, turn.owner, sampled=True)
            following = turns[index + 1].role if index + 1 < len(turns) else None
            if turn.role == "user" and (
                following == "assistant" or following is None and add_generation_prompt
            ):
                opens_reasoning = thinking and (not drop or index >= last_turn)
                writer.write(ASSISTANT + (THINK_START if opens_reasoning else THINK_END), -1)
        return self._tokens(writer, messages)

    @staticmethod
    def _user(writer: _Writer, turn: _Turn) -> None:
        writer.write(USER, turn.owner)
        for number, block in enumerate(turn.blocks):
            if number:
                writer.write("\n\n", block.owner)
            writer.pieces(block.pieces, block.owner)

    def _tokens(self, writer: _Writer, messages, prefix: Sequence[int] = ()) -> RenderedTokens:
        """Tokenize the rendered text in one pass, as the template's text is tokenized."""
        text, images = writer.text(self._image_count)
        encoding = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        owners, sampled = writer.attribute(encoding["offset_mapping"])
        ids = [*prefix, *encoding["input_ids"]]
        return RenderedTokens(
            token_ids=ids,
            message_indices=[-1] * len(prefix) + owners,
            sampled_mask=[False] * len(prefix) + sampled,
            is_content=[],
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=extract_message_tool_names(list(messages)),
            multi_modal_data=_images(ids, self._image, images),
        )

    def _image_count(self, part: Mapping[str, Any]) -> int:
        from PIL import Image

        from loops_cookbook.llm.images import image_bytes

        raw, _ = image_bytes(part)
        key = hashlib.sha256(raw).hexdigest()
        if key not in self._image_cache:
            with Image.open(io.BytesIO(raw)) as image:
                self._image_cache[key] = image_tokens(*image.size)
        return self._image_cache[key]

    # --- continuing a sampled turn ----------------------------------------------------

    def prepare_extension(self, previous_messages, new_messages, tools):
        """Sort the new tool results into the order of the previous assistant's calls.

        The template sorts results by the calls of the latest assistant message that made
        any. The bridge sees only the new messages, so this reads those calls from the
        history. It never renders the history again.
        """
        call_ids: list[Any] = []
        for message in previous_messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                call_ids = _call_ids(message)
        return _sorted_results(
            list(new_messages),
            call_ids,
            lambda m: m.get("tool_call_id", "") if m.get("role") == "tool" else None,
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
        previous_multi_modal_data=None,
    ) -> RenderedTokens | None:
        """Extend the sampled tokens with the new messages, or return ``None`` to render in full.

        The new messages must be user and tool messages. The bridge declines when a full
        render would drop the sampled reasoning, which happens in thinking mode with
        ``drop_thinking`` on, no tools, and a new message that starts a turn.
        """
        if not previous_prompt_ids or not new_messages:
            return None
        if any(message.get("role") not in ("user", "tool") for message in new_messages):
            return None
        try:
            (turn,) = _merge(new_messages)
        except (ValueError, TypeError):
            return None
        config = self.config
        if (
            config.enable_thinking
            and config.drop_thinking
            and not tools
            and self._starts_turn(turn)
        ):
            return None
        completion = list(previous_completion_ids)
        parsed = self.parse_response(completion, prompt_ids=list(previous_prompt_ids), tools=tools)
        if parsed.reasoning_complete is False:
            if self._eos in completion:
                return None
            completion.append(self._think_end)
        prefix = trim_to_turn_close(
            list(previous_prompt_ids), completion, {self._eos}, synthesize_close=self._eos
        )
        if prefix is None:
            return None
        writer = _Writer()
        self._user(writer, turn)
        # A continuation always ends at or after the last turn, so reasoning opens.
        writer.write(ASSISTANT + (THINK_START if config.enable_thinking else THINK_END), -1)
        try:
            return self._tokens(writer, new_messages, prefix)
        except (ValueError, TypeError):
            return None

    # --- parsing ----------------------------------------------------------------------

    def parse_response(self, token_ids, *, tools=None, prompt_ids=None) -> ParsedResponse:
        """Read a completion, and read each tool argument back as the template wrote it.

        The template marks each argument with ``string="true"`` or ``string="false"``. A
        string comes back exactly as written. Any other value comes back as its JSON, or
        as the text when it isn't JSON, for the tool to judge.
        """
        ids = list(token_ids)
        if self._eos in ids:
            ids = ids[: ids.index(self._eos)]
        thinking = (
            prompt_ids[-1] == self._think_start if prompt_ids else self.config.enable_thinking
        )
        scanned = scan_tagged_response(
            self._tokenizer,
            ids,
            thinking=thinking,
            atomic_markers={self._think_start: "think_start", self._think_end: "think_end"},
            grammar=_GRAMMAR,
        )
        content = scanned.content
        if scanned.tools and content.endswith("\n\n"):
            # The template writes a blank line between the content and the calls.
            content = content[:-2]
        declared = {tool.get("function", tool).get("name") for tool in tools or ()}
        return ParsedResponse(
            content=content,
            reasoning_content=scanned.boundary.text,
            tool_calls=[_read_call(block, declared) for block in scanned.tools],
            reasoning_complete=not scanned.boundary.is_open,
        )


def _read_call(block, declared: set[Any]) -> ParsedToolCall:
    """Read one ``invoke`` block whole, or report why it can't be read."""
    raw = block.text
    header = _HEADER.match(raw)
    name = header.group(1) if header else None
    end = f"</{_INVOKE}>"
    body = raw[header.end() : len(raw) - len(end) if block.closed else len(raw)] if header else ""
    arguments: dict[str, Any] = {}
    # A quote in a name, or a tag opener in a value, means that a lazy match ran past
    # an unclosed or misspelled tag into the next one, so the call can't be read whole.
    malformed = header is None or '"' in (name or "")
    rest, cursor = [], 0
    for match in _PARAMETER_BLOCK.finditer(body):
        rest.append(body[cursor : match.start()])
        cursor = match.end()
        key, is_string, value = match.groups()
        if not key or '"' in key or key in arguments or f"<{_DSML}" in value:
            malformed = True
        elif is_string == "true":
            arguments[key] = value
        else:
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                arguments[key] = value
    rest.append(body[cursor:])
    # The template writes the parameters one per line, so only newlines may remain.
    if "".join(rest).strip("\n"):
        malformed = True
    if not block.closed:
        status = ToolCallParseStatus.UNCLOSED_BLOCK
    elif not name:
        status = ToolCallParseStatus.MISSING_NAME
    elif malformed:
        status = ToolCallParseStatus.MALFORMED_STRUCTURE
    elif declared and name not in declared:
        status = ToolCallParseStatus.UNKNOWN_TOOL
    else:
        status = ToolCallParseStatus.OK
    return ParsedToolCall(
        raw=raw, name=name, arguments=arguments, token_span=block.token_span, status=status
    )


class _Writer:
    """Collects the rendered text with each character's owner and generated flag."""

    def __init__(self) -> None:
        self._spans: list[tuple[Piece, int, bool]] = []
        self._starts: list[int] = []
        self._owners: list[tuple[int, bool]] = []

    def write(self, text: str, owner: int, *, sampled: bool = False) -> None:
        if text:
            self._spans.append((text, owner, sampled))

    def pieces(self, pieces: Sequence[Piece], owner: int, *, sampled: bool = False) -> None:
        for piece in pieces:
            if not isinstance(piece, str) or piece:
                self._spans.append((piece, owner, sampled))

    def text(self, count) -> tuple[str, list[Mapping[str, Any]]]:
        """Return the text, with each image expanded to its tokens, and the images in order."""
        parts, images, position = [], [], 0
        for piece, owner, sampled in self._spans:
            if not isinstance(piece, str):
                images.append(piece)
                piece = IMAGE * count(piece)
            parts.append(piece)
            self._starts.append(position)
            self._owners.append((owner, sampled))
            position += len(piece)
        return "".join(parts), images

    def attribute(self, offsets) -> tuple[list[int], list[bool]]:
        """Give each token the owner and flag of the text where it starts."""
        owners, sampled = [], []
        for start, _ in offsets:
            owner, flag = self._owners[max(bisect.bisect_right(self._starts, start) - 1, 0)]
            owners.append(owner)
            sampled.append(flag)
        return owners, sampled


def v41(tokenizer, chat_template_kwargs):
    """Build the renderer for DeepSeek-V4.1-Flash, which the cookbook selects for it."""
    return _wrap(tokenizer, "v4.1", chat_template_kwargs)


def v41_1017(tokenizer, chat_template_kwargs):
    """Build the renderer for DeepSeek-V4.1-Flash-1017, which the cookbook selects for it."""
    return _wrap(tokenizer, "1017", chat_template_kwargs)


def _wrap(tokenizer, variant: Variant, chat_template_kwargs):
    from loops_cookbook.llm.prime import PrimeRenderer

    config = DeepSeekV41RendererConfig(**dict(chat_template_kwargs))
    built = DeepSeekV41Renderer(tokenizer, variant, config)
    return PrimeRenderer(built, prepare_extension=built.prepare_extension)
