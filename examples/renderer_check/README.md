# Renderer check

A live check that a renderer passes tool-call arguments to the agent exactly as the
model wrote them, and that a turn with an unusual call still continues the episode.
Run it when a renderer is added or its parsing changes.

The agent (`episode.py`) edits a small in-memory project with tools whose schemas
cover every kind of argument the chat templates write differently:

| Case | Tool |
| --- | --- |
| Strings with leading or trailing whitespace and inner newlines | `edit_file.old`, `edit_file.new`, `write_file.content` |
| Integer, number, boolean, null | `configure`, `set_setting` |
| Enum | `configure.level` |
| Union with a string branch | `set_setting.value` |
| Array of strings | `tag_file.tags` |
| Array of objects through `$defs` | `apply_patch.ops` |
| A tool the schema does not declare | `delete_file`, mentioned in the prompt on the `delete-*` tasks |

Every tool validates strictly, so a string `"3"` for an integer fails the call and the
model sees the failure. `grade()` compares the whole of the files, settings and tags
with the task's expected state, byte for byte and type for type. A `delete-*` task
expects `old.py` left in place: no tool can remove it, and the right answer is to say
so rather than to empty or overwrite it.

`tasks.jsonl` holds the tasks. A task that asks for a numeric string under
`set_setting.value` cannot be expressed: the templates write a string verbatim, so
`2024` reads back as the integer the schema also accepts. That is a property of the
format, not of a renderer, so no task asks for it.

## Running

One config per model. Serverless where the model is served that way, dedicated spot
otherwise; each ends after two updates or thirty groups without variance.

    uv run loops check examples/renderer_check/glm53_flash.toml
    uv run loops run examples/renderer_check/glm53_flash.toml
    uv run python examples/renderer_check/report.py runs/renderer-check/<run dir>

The report reads the run's transcripts and prints, per tool and argument, the status
of every call with whether its native message was kept, how each argument arrived
(verbatim, stripped, changed, or typed), and the reward per task. A good run shows
every string `verbatim`, every typed value `typed <kind>`, `unknown_tool` calls with
`kept=True`, and no `STRIPPED`, `CHANGED`, `DROPPED` or `MISMATCH` lines.

To check another renderer, copy a config, change `base_model` and the
`renderer_chat_template_kwargs` the model takes, and run it.
