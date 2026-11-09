# Contributing

Open pull requests (PRs) against `main`. CI must pass before a PR merges, and a code
owner from [.github/CODEOWNERS](.github/CODEOWNERS) must approve it.

## Commits and PR titles

`main` takes squash merges, so a PR's title becomes its commit there. Write the
title as a [Conventional Commit](https://www.conventionalcommits.org/):

```text
type: summary
```

- `type` is one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
  `deps`, `chore` or `revert`.
- Leave out the scope. Most changes touch more than one area, and the summary can name
  the area when it matters.
- Write the summary in the imperative mood, as an instruction to the code, such as
  `fix: read GLM-5.3 tool arguments as the chat template wrote them`.
- Start the summary with a letter or a digit. Don't end it with a period.
- Mark a breaking change with `!`, such as `feat!: rename max_steps`.

The **pr-title** check warns about a title in any other form, but it doesn't block the
merge. The **pr-labels** workflow labels each PR from the type in its title, and the
release notes group the PRs by those labels.

The pre-commit hooks check each commit message in the same way. They also run ruff and
some checks on file contents, such as trailing whitespace and valid TOML. Install the
hooks once per clone:

```sh
uvx pre-commit install
```

## Development

Install the development dependencies and run the checks:

```sh
uv sync --locked --extra harbor-blaxel --extra harbor-daytona --extra wandb
uv run python scripts/generate_openai_models.py --check
uv run ruff check src tests patterns examples tasks scripts
uv run ruff format --check src tests patterns examples tasks scripts
uv run pyright
uv run pytest -q
```

Tests that need real tokenizers skip unless one of these variables points at a Hugging
Face cache: `LOOPS_GLM_TOKENIZER_CACHE`, `LOOPS_QWEN_TOKENIZER_CACHE`,
`LOOPS_NEMOTRON_TOKENIZER_CACHE` or `LOOPS_MUSE_TOKENIZER_CACHE`. Alternatively,
`LOOPS_DOWNLOAD_CONTRACT_MODELS=1` downloads them. Live tests against real Blaxel
sandboxes and Loops never run by default. See [tests/README.md](tests/README.md).

## Releasing

All work lands on `main`, by squash merge. A release is a tag on `main`, such as
`v0.3.0`, and the version comes from that tag. Nothing in the tree names the version.

To release what is on `main`:

1. If the release moves the `blaxel` range in `pyproject.toml`, run the live Blaxel
   tests that [tests/README.md](tests/README.md) describes. Pyright and the unit tests
   can't show how the SDK behaves against the real service. Daytona has no live tests
   yet, so check a move of the `daytona` range by hand.
2. Run the **release** workflow on `main` with the version, such as `0.4.0`. To run
   it, go to Actions → release → Run workflow. You can also run
   `gh workflow run release.yml -r main -f version=0.4.0`.
3. The workflow checks that the version is new and above the last release. It tags the
   head of `main` and runs lint, pyright and the tests against `uv.lock`. Then it
   builds the wheel and the sdist, and checks that they carry the tag's version. At the
   same time it tests both ends of the dependency ranges, which is the
   **dependency-ranges** check.
4. Approve the **pypi** deployment. This approval is the review of the release. The
   upload waits for the build and for both dependency-ranges checks. It publishes the
   wheel and the sdist to PyPI. Then the workflow pushes the tag and publishes a GitHub
   release with the files attached. The release also lists the dependency versions of
   both checks, and the notes since the last release.

The PyPI page shows the README without the repository around it, so `pyproject.toml`
turns the README's relative links into links to `main`. The release workflow points
those links at the release's tag instead, so the PyPI page shows the released files.

The **dependency-ranges** workflow also runs each night. It resolves the newest versions
that the ranges allow, and the floor of each direct dependency.

Like `uv.lock`, the check skips releases younger than 7 days, through `exclude-newer` in
`pyproject.toml`. A broken release is often yanked within that time. Baseten's own
packages, `baseten`, `baseten-bdn` and `baseten-loops`, are exempt, so the cookbook can
move to their releases on the day they ship. Add a new Baseten dependency to that list.
To use another package's release before it is 7 days old, add the package to
`exclude-newer-package` with a cutoff date. Add a note to delete the entry once the 7
days pass.

When the check fails on the newest versions, an upstream release has broken the
cookbook for users who install it. Fix the code, or lower the cap. When the check fails
on the floors, raise the floor.

A release is always the head of `main`. To fix a bug in a release, merge the fix into
`main`, then release `main` with the next version.

To publish a pre-release, push a pre-release tag, such as with
`git tag v0.4.0-alpha.1 && git push origin v0.4.0-alpha.1`. The tag must name a commit
on `main`, because an upload to PyPI is permanent. That tag builds as version
`0.4.0a1`, and it is published as a pre-release. A pushed `vX.Y.Z` tag is refused,
because a release comes only from the release workflow.

## Diagrams

Most diagrams in [docs/assets](docs/assets/) have a source, the `.excalidraw` file of
the same name. To change a diagram, open its file at
[excalidraw.com](https://excalidraw.com), edit it and save it back. Then export it
twice, with "Embed scene" and the background turned on. Export it once in the light
theme, over `<name>.svg`, and once in the dark theme, over `<name>-dark.svg`.
