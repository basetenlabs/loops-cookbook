"""The cookbook's renderer for DeepSeek-V4.1-Flash and DeepSeek-V4.1-Flash-1017.

It follows each checkpoint's ``chat_template.jinja``: DeepSeek-V4.1-Flash at revision
314113f8a7dadb388215396ba5fbee2c610a04d6, and DeepSeek-V4.1-Flash-1017 at revision
08c8657f02291e11b6f6b63e0e16014f8e748518. Prime 0.2.0 has a renderer for V4 only. V4.1
renamed the tool-call tags, opens the prompt with ``<｜System｜>``, writes the reasoning
effort as a number and takes images. So this module renders the template itself, and it
uses Prime only for the result types, the turn-close helper and the response scanner.

The template writes one ``<｜deepseek_image｜>`` per image, and the inference engine
expands it. The trainer needs the expanded form, so this renderer writes the token once
for each image token. The checkpoints ship no image processor. The count comes from the
image's size, as the trainer's ``deepseek_v41_image_tokens`` computes it.

The two checkpoints differ in when a turn starts, which decides whose reasoning the
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
from dataclasses import dataclass, field
from typing import Any, Literal

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

_EFFORT_BUDGETS: dict[Variant, dict[str, int]] = {
    "v4.1": {"low": 50, "medium": 62, "high": 75, "max": 100},
    "1017": {
        "none": 75,
        "minimal": 50,
        "low": 50,
        "medium": 75,
        "high": 75,
        "xhigh": 75,
        "max": 100,
    },
}
_OPTIONS: dict[Variant, frozenset[str]] = {
    "v4.1": frozenset({"thinking_mode", "enable_thinking", "reasoning_effort", "drop_thinking"}),
    "1017": frozenset(
        {"thinking_mode", "thinking", "enable_thinking", "reasoning_effort", "drop_thinking"}
    ),
}

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


@dataclass(frozen=True)
class Options:
    """The chat-template options, resolved as the template resolves them."""

    thinking: bool
    drop_thinking: bool
    budget: int


def options(variant: Variant, kwargs: Mapping[str, Any]) -> Options:
    """Resolve ``renderer_chat_template_kwargs`` as the checkpoint's template does.

    An option the template doesn't read is refused, so a misspelled option fails at
    ``loops check`` instead of being ignored.
    """
    unknown = sorted(set(kwargs) - _OPTIONS[variant])
    if unknown:
        raise ValueError(
            f"DeepSeek-V4.1 renderer options are {sorted(_OPTIONS[variant])}, not {unknown}"
        )
    effort = kwargs.get("reasoning_effort")
    mode = kwargs.get("thinking_mode")
    if mode is None:
        switch = kwargs.get("thinking") if variant == "1017" else None
        if switch is None:
            switch = kwargs.get("enable_thinking")
        if switch is not None:
            mode = "thinking" if switch else "chat"
        else:
            mode = "chat" if variant == "1017" and effort == "none" else "thinking"
    if mode not in ("chat", "thinking"):
        raise ValueError(f'thinking_mode must be "chat" or "thinking", not {mode!r}')
    if effort is None:
        effort = "high"
    budgets = _EFFORT_BUDGETS[variant]
    if isinstance(effort, str) and effort in budgets:
        budget = budgets[effort]
    elif isinstance(effort, int) and not isinstance(effort, bool) and 1 <= effort <= 100:
        budget = effort
    else:
        raise ValueError(
            f"reasoning_effort must be an integer from 1 to 100 or one of {sorted(budgets)}, "
            f"not {effort!r}"
        )
    drop = kwargs.get("drop_thinking", True)
    if not isinstance(drop, bool):
        raise TypeError(f"drop_thinking must be true or false, not {drop!r}")
    return Options(thinking=mode == "thinking", drop_thinking=drop, budget=budget)


# --- messages as the template merges them ---------------------------------------------

Piece = str | Mapping[str, Any]
"""Text, or an image part, in the order the template writes them."""


@dataclass
class _Block:
    kind: Literal["text", "image", "tool_result"]
    pieces: list[Piece]
    owner: int
    call_id: Any = ""


@dataclass
class _Turn:
    role: str
    owner: int
    blocks: list[_Block] = field(default_factory=list)
    message: Mapping[str, Any] = field(default_factory=dict)
    tools: Sequence[Mapping[str, Any]] = ()


def _is_image(part: Any) -> bool:
    from loops_cookbook.llm.images import is_image_part

    return isinstance(part, Mapping) and is_image_part(part)


def _text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("DeepSeek-V4.1 text must be a string")
    if IMAGE in text:
        raise ValueError(f"text contains {IMAGE}; pass images as image parts")
    return text


def _joined(content: Any, *, images: bool) -> list[Piece]:
    """Return a content list's text and images, joined by blank lines as the template joins them."""
    pieces: list[Piece] = []
    for part in content:
        if pieces:
            pieces.append("\n\n")
        if isinstance(part, Mapping) and part.get("type") == "text":
            pieces.append(_text(part.get("text") or ""))
        elif images and _is_image(part):
            pieces.append(part)
        elif _is_image(part):
            raise ValueError("DeepSeek-V4.1 takes images in system, user and tool messages only")
        else:
            raise ValueError("DeepSeek-V4.1 content parts must be text or images")
    return pieces


def _body(content: Any, *, images: bool) -> list[Piece]:
    if content is None:
        return [""]
    if isinstance(content, str):
        return [_text(content)]
    if isinstance(content, list):
        return _joined(content, images=images)
    raise ValueError("DeepSeek-V4.1 content must be a string or a list of parts")


_REFUSED_KEYS = ("task", "wo_eos", "response_format", "tools", "content_blocks")


def _merge(
    variant: Variant, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
) -> list[_Turn]:
    """Merge tool results and consecutive user messages into user turns, as the template does."""
    turns: list[_Turn] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        refused = [key for key in _REFUSED_KEYS if message.get(key) is not None]
        if refused:
            raise ValueError(f"the DeepSeek-V4.1 renderer does not render {refused[0]!r}")
        if role == "tool":
            content = message.get("content")
            if isinstance(content, list):
                pieces = _joined(content, images=True)
            elif content is None:
                # The V4.1 template writes a missing tool content as Jinja writes None.
                pieces = ["" if variant == "1017" else "None"]
            else:
                pieces = [_text(content)]
            block = _Block(
                "tool_result",
                ["<tool_result>", *pieces, "</tool_result>"],
                index,
                message.get("tool_call_id", ""),
            )
            if turns and turns[-1].role == "user":
                turns[-1].blocks.append(block)
            else:
                turns.append(_Turn("user", index, [block]))
        elif role == "user":
            content = message.get("content")
            if isinstance(content, list):
                blocks = []
                for part in content:
                    if _is_image(part):
                        blocks.append(_Block("image", [part], index))
                    elif isinstance(part, Mapping) and part.get("type") == "text":
                        blocks.append(_Block("text", [_text(part.get("text") or "")], index))
                    else:
                        raise ValueError("DeepSeek-V4.1 content parts must be text or images")
            else:
                blocks = [_Block("text", [_text(content or "")], index)]
            if turns and turns[-1].role == "user":
                turns[-1].blocks.extend(blocks)
            else:
                turns.append(_Turn("user", index, blocks))
        elif role in ("system", "assistant"):
            if role == "assistant":
                _reasoning(message)  # The template checks reasoning it then drops, too.
            turns.append(_Turn(role, index, message=message))
        else:
            raise ValueError(f"the DeepSeek-V4.1 template has no {role!r} role")
    if tools:
        if turns and turns[0].role == "system":
            turns[0].tools = tools
        else:
            turns.insert(0, _Turn("system", -1, message={"content": ""}, tools=tools))
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


def _sort_results(turns: list[_Turn]) -> None:
    """Sort each user turn's tool results into the order of the latest assistant's calls."""
    call_ids: list[Any] = []
    for turn in turns:
        if turn.role == "assistant" and turn.message.get("tool_calls"):
            call_ids = _call_ids(turn.message)
        elif turn.role == "user":
            results = [block for block in turn.blocks if block.kind == "tool_result"]
            if len(results) < 2:
                continue
            ordered = iter(sorted(results, key=lambda block: _order(call_ids, block.call_id)))
            turn.blocks = [
                next(ordered) if block.kind == "tool_result" else block for block in turn.blocks
            ]


def _qualified_name(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError("a tool name must be a string")
    parts = name.split("::", 1)
    if len(parts) == 2 and "::" in parts[1]:
        raise ValueError(f"Tool name must not contain '::': {parts[1]}")
    return name


def _tool_schemas(tools: Sequence[Mapping[str, Any]]) -> str:
    schemas = []
    for tool in tools:
        function = tool.get("function", tool)
        if tool.get("namespace") is not None or function.get("namespace") is not None:
            raise ValueError("the DeepSeek-V4.1 renderer does not render tool namespaces")
        schema = dict(function)
        schema["name"] = _qualified_name(function.get("name"))
        schemas.append(json.dumps(schema, ensure_ascii=False))
    return _TOOLS_HEADER + "\n".join(schemas) + _TOOLS_FOOTER


def _tool_calls(variant: Variant, calls: Sequence[Mapping[str, Any]]) -> str:
    invokes = []
    for call in calls:
        function = call.get("function") or {}
        if call.get("namespace") or function.get("namespace"):
            raise ValueError("the DeepSeek-V4.1 renderer does not render tool namespaces")
        arguments = function.get("arguments")
        if isinstance(arguments, str) and variant == "1017":
            raise ValueError("Tool call arguments must be a dict, not a JSON string")
        if not isinstance(arguments, Mapping):
            arguments = {"arguments": arguments}
        lines = [
            f'<{_PARAMETER} name="{key}" string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
            f"</{_PARAMETER}>"
            for key, value in arguments.items()
        ]
        name = _qualified_name(function.get("name"))
        invokes.append(f'<{_INVOKE} name="{name}">\n' + "\n".join(lines) + f"\n</{_INVOKE}>")
    return f"\n\n<{_CALLS}>\n" + "\n".join(invokes) + f"\n</{_CALLS}>"


def _reasoning(message: Mapping[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("thinking")
    return _text(reasoning or "")


# --- the renderer ---------------------------------------------------------------------


class DeepSeekV41Renderer:
    """Renders DeepSeek-V4.1-Flash and -1017 conversations as each one's chat template does."""

    reads_calls_whole = True  # a block is read whole or reported, never in part
    is_multimodal = True  # renderers.base.is_multimodal honours this.

    def __init__(self, tokenizer, variant: Variant, chat_template_kwargs: Mapping[str, Any]):
        self._tokenizer = tokenizer
        self.variant: Variant = variant
        self.options = options(variant, chat_template_kwargs)
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

    # --- rendering --------------------------------------------------------------------

    def render(self, messages, *, tools=None, add_generation_prompt=False) -> RenderedTokens:
        return self._render(messages, tools or (), add_generation_prompt, continuation=False)

    def _render(self, messages, tools, add_generation_prompt, *, continuation) -> RenderedTokens:
        """Render as the template does.

        A ``continuation`` renders messages that follow a sampled assistant turn. It
        has no beginning of sentence and no effort prompt, and a system message in it
        is not the conversation's first.
        """
        if not messages:
            raise ValueError("No messages provided.")
        # A continuation's tools are already declared at the start of the conversation.
        turns = _merge(self.variant, messages, () if continuation else tools)
        _sort_results(turns)
        thinking = self.options.thinking
        drop = self.options.drop_thinking and not tools
        last_turn = -1
        for index, turn in enumerate(turns):
            if self._starts_turn(turn, first=index == 0 and not continuation):
                last_turn = index

        writer = _Writer()
        if not continuation:
            writer.write(BOS, -1)
        for index, turn in enumerate(turns):
            first = index == 0 and not continuation
            if first and (thinking or turn.role == "system"):
                writer.write(SYSTEM, -1)
            if first and thinking:
                writer.write(
                    f"Reasoning Effort: {self.options.budget} (range 1-100, the higher the "
                    "value, the more thorough the reasoning)\n\n",
                    -1,
                )
            if turn.role == "system":
                if not first:
                    writer.write(SYSTEM, turn.owner)
                writer.pieces(_body(turn.message.get("content"), images=True), turn.owner)
                if turn.tools:
                    writer.write("\n\n" + _tool_schemas(turn.tools), turn.owner)
            elif turn.role == "user":
                writer.write(USER, turn.owner)
                for number, block in enumerate(turn.blocks):
                    if number:
                        writer.write("\n\n", block.owner)
                    writer.pieces(block.pieces, block.owner)
            else:
                message = turn.message
                if thinking and (not drop or index > last_turn):
                    writer.write(_reasoning(message) + THINK_END, turn.owner, sampled=True)
                writer.pieces(_body(message.get("content"), images=False), turn.owner, sampled=True)
                if message.get("tool_calls"):
                    writer.write(
                        _tool_calls(self.variant, message["tool_calls"]), turn.owner, sampled=True
                    )
                writer.write(EOS, turn.owner, sampled=True)
            last = index == len(turns) - 1
            if not last and turns[index + 1].role != "assistant":
                continue
            opens_turn = turn.role == "user" or (turn.role == "system" and not first)
            if opens_turn and (add_generation_prompt or not last):
                opens_reasoning = thinking and (not drop or index >= last_turn)
                writer.write(ASSISTANT + (THINK_START if opens_reasoning else THINK_END), -1)
        return self._tokens(writer, messages)

    def _starts_turn(self, turn: _Turn, *, first: bool) -> bool:
        """Whether a merged message starts a turn, which bounds the reasoning the template drops."""
        if turn.role == "system":
            return not first
        if turn.role != "user":
            return False
        return self.variant == "v4.1" or any(block.kind != "tool_result" for block in turn.blocks)

    def _tokens(self, writer: _Writer, messages) -> RenderedTokens:
        """Tokenize the rendered text in one pass, as the template's text is tokenized."""
        text, images = writer.text(self._image_count)
        encoding = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoding["input_ids"])
        owners, sampled = writer.attribute(encoding["offset_mapping"])
        ranges, at = [], 0
        while at < len(ids):
            if ids[at] == self._image:
                start = at
                while at < len(ids) and ids[at] == self._image:
                    at += 1
                ranges.append(PlaceholderRange(offset=start, length=at - start))
            else:
                at += 1
        if len(ranges) != len(images):
            raise ValueError(f"rendered {len(images)} images but found {len(ranges)} image runs")
        data = None
        if ranges:
            data = MultiModalData(
                mm_hashes={"image": [f"deepseek-{r.offset}" for r in ranges]},
                mm_placeholders={"image": ranges},
                mm_items={"image": [dict(part) for part in images]},
            )
        return RenderedTokens(
            token_ids=ids,
            message_indices=owners,
            sampled_mask=sampled,
            is_content=[],
            message_roles=[message.get("role") or "" for message in messages],
            message_tool_names=extract_message_tool_names(list(messages)),
            multi_modal_data=data,
        )

    def _image_count(self, part: Mapping[str, Any]) -> int:
        from PIL import Image

        from loops_cookbook.llm.images import image_bytes

        raw, _ = image_bytes(part)
        key = hashlib.sha256(raw).hexdigest()
        if key not in self._image_cache:
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
            self._image_cache[key] = image_tokens(width, height)
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
        result = list(new_messages)
        start = 0
        while start < len(result):
            end = start
            while end < len(result) and result[end].get("role") in ("user", "tool"):
                end += 1
            run = result[start:end]
            results = [message for message in run if message.get("role") == "tool"]
            if len(results) > 1:
                ordered = iter(
                    sorted(results, key=lambda m: _order(call_ids, m.get("tool_call_id", "")))
                )
                result[start:end] = [
                    next(ordered) if message.get("role") == "tool" else message for message in run
                ]
            start = end + 1
        return result

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

        The bridge declines when a full render would drop the sampled reasoning. That
        happens when thinking is on, ``drop_thinking`` is on, no tools are declared and a
        new message starts a turn.
        """
        if not previous_prompt_ids or not new_messages:
            return None
        if any(message.get("role") not in ("user", "tool", "system") for message in new_messages):
            return None
        if self.options.thinking and self.options.drop_thinking and not tools:
            merged = _merge(self.variant, new_messages, ())
            if any(self._starts_turn(turn, first=False) for turn in merged):
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
        try:
            extension = self._render(new_messages, tools or (), True, continuation=True)
        except (ValueError, TypeError):
            return None
        ids = prefix + extension.token_ids
        prior = len(prefix)
        data = None
        new_images = (
            extension.multi_modal_data.mm_items["image"] if extension.multi_modal_data else []
        )
        ranges = [
            PlaceholderRange(offset=offset, length=length)
            for offset, length in _runs(prefix, self._image)
        ] + [
            PlaceholderRange(offset=prior + r.offset, length=r.length)
            for r in (
                extension.multi_modal_data.mm_placeholders["image"]
                if extension.multi_modal_data
                else []
            )
        ]
        if ranges:
            data = MultiModalData(
                mm_hashes={"image": [f"deepseek-{r.offset}" for r in ranges]},
                mm_placeholders={"image": ranges},
                mm_items={"image": [{} for _ in range(len(ranges) - len(new_images))] + new_images},
            )
        return RenderedTokens(
            token_ids=ids,
            message_indices=[-1] * prior + extension.message_indices,
            sampled_mask=[False] * len(ids),
            is_content=[],
            message_roles=extension.message_roles,
            message_tool_names=extension.message_tool_names,
            multi_modal_data=data,
        )

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
        thinking = prompt_ids[-1] == self._think_start if prompt_ids else self.options.thinking
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
        declared = {
            tool.get("function", tool).get("name")
            for tool in tools or ()
            if isinstance(tool, Mapping)
        }
        return ParsedResponse(
            content=content,
            reasoning_content=scanned.boundary.text,
            tool_calls=[_read_call(block, declared) for block in scanned.tools],
            reasoning_complete=not scanned.boundary.is_open,
        )


def _runs(ids: Sequence[int], token: int) -> list[tuple[int, int]]:
    runs, at = [], 0
    while at < len(ids):
        if ids[at] == token:
            start = at
            while at < len(ids) and ids[at] == token:
                at += 1
            runs.append((start, at - start))
        else:
            at += 1
    return runs


def _read_call(block, declared: set[Any]) -> ParsedToolCall:
    """Read one ``invoke`` block whole, or report why it can't be read."""
    raw = block.text
    header = _HEADER.match(raw)
    name = header.group(1) if header else None
    end = f"</{_INVOKE}>"
    body = raw[header.end() : len(raw) - len(end) if block.closed else len(raw)] if header else ""
    arguments: dict[str, Any] = {}
    malformed = header is None
    rest, cursor = [], 0
    for match in _PARAMETER_BLOCK.finditer(body):
        rest.append(body[cursor : match.start()])
        cursor = match.end()
        key, is_string, value = match.groups()
        if key in arguments:
            malformed = True
            continue
        if is_string == "true":
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

    def write(self, text: str, owner: int, *, sampled: bool = False) -> None:
        if text:
            self._spans.append((text, owner, sampled))

    def pieces(self, pieces: Sequence[Piece], owner: int, *, sampled: bool = False) -> None:
        for piece in pieces:
            if isinstance(piece, str):
                self.write(piece, owner, sampled=sampled)
            else:
                self._spans.append((piece, owner, sampled))

    def text(self, count) -> tuple[str, list[Mapping[str, Any]]]:
        """Return the text, with each image expanded to its tokens, and the images in order."""
        parts, images, self._starts, self._owners = [], [], [], []
        position = 0
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
            span = max(bisect.bisect_right(self._starts, start) - 1, 0)
            owner, flag = self._owners[span]
            owners.append(owner)
            sampled.append(flag)
        return owners, sampled


def v41(tokenizer, chat_template_kwargs):
    """Build the renderer for DeepSeek-V4.1-Flash, which the cookbook selects for it."""
    return _wrap(DeepSeekV41Renderer(tokenizer, "v4.1", chat_template_kwargs))


def v41_1017(tokenizer, chat_template_kwargs):
    """Build the renderer for DeepSeek-V4.1-Flash-1017, which the cookbook selects for it."""
    return _wrap(DeepSeekV41Renderer(tokenizer, "1017", chat_template_kwargs))


def _wrap(built: DeepSeekV41Renderer):
    from loops_cookbook.llm.prime import PrimeRenderer

    return PrimeRenderer(built, prepare_extension=built.prepare_extension)
