# Contributing

Open pull requests against `develop`; CI must pass before merge.

## Commits and PR titles

`develop` takes squash merges, so a PR's title becomes its commit there. Write the
title as a [Conventional Commit](https://www.conventionalcommits.org/):

```text
type(scope): summary
```

- `type` is one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
  `deps`, `chore` or `revert`.
- `scope` is optional. Use the area you changed, such as `rl`, `sft`, `gateway` or
  `examples`.
- Start the summary with a letter or a digit. Do not end it with a period.
- Mark a breaking change with `!`, such as `feat(rl)!: rename max_steps`.

The **pr-title** check gives a warning for other titles, but does not block the
merge. PRs into `main` are releases and keep the `Release vX.Y.Z` title (see [Releasing](#releasing)).

The pre-commit hooks check each commit message the same way, and run ruff and some
file checks. Install them once per clone:

```sh
uvx pre-commit install
```

## Development

```sh
uv sync --locked --extra harbor-blaxel --extra harbor-daytona --extra wandb
uv run python scripts/generate_openai_models.py --check
uv run ruff check src tests patterns examples tasks scripts
uv run ruff format --check src tests patterns examples tasks scripts
uv run pyright
uv run pytest -q
```

Tests that need real tokenizers skip unless `LOOPS_GLM_TOKENIZER_CACHE`,
`LOOPS_QWEN_TOKENIZER_CACHE`, `LOOPS_NEMOTRON_TOKENIZER_CACHE` or `LOOPS_MUSE_TOKENIZER_CACHE` point at a Hugging Face
cache, or `LOOPS_DOWNLOAD_CONTRACT_MODELS=1` downloads them. Live tests against real
Blaxel sandboxes and Loops never run by default; see [tests/README.md](tests/README.md).

## Releasing

`develop` is where work lands, by squash merge. `main` holds only released code: every
PR into it is a release, from `develop` or a `hotfix/` branch, and merges with a merge
commit (never squash or rebase), so `main` and `develop` keep one history and the next
release PR carries only new commits. The version comes from the git tag; only the
README's install lines name it.

To release what is on `develop`:

1. If the release moves the `blaxel` range in `pyproject.toml`, run the live Blaxel
   tests ([tests/README.md](tests/README.md)). Pyright and the unit tests cannot show
   how the SDK behaves against the real service. Daytona has no live tests yet, so
   check a `daytona` move by hand.
2. On `develop`, change the tag in the README's `uv add` lines to the new version.
3. Run the **prepare-release** workflow (Actions → prepare-release → Run workflow) with
   the version, such as `0.2.0`. It checks the version is new and above the last
   release, and opens a PR from `develop` into `main` titled "Release v0.2.0" with a
   preview of the release notes. (`gh pr create -B main -H develop -t "Release v0.2.0"`
   does the same by hand.)
4. CI on a PR the workflow opened waits for a maintainer: **Approve and run** it.
5. Merge it with **Create a merge commit**. The **release** workflow tags the merge
   commit and runs lint, pyright, and the tests, against `uv.lock` and against both
   ends of the ranges (**dependency-ranges**). It builds the wheel and sdist
   and checks they carry the tag's version. Then it publishes a GitHub release with
   them attached, the dependency versions of both checks, and notes since the last
   release.

The **dependency-ranges** workflow also runs each night. It resolves the newest
versions the ranges allow, and the floor of each direct dependency. Like `uv.lock`, it
skips releases younger than 7 days (`exclude-newer` in `pyproject.toml`), because a
broken release is often yanked in that time. Baseten's own packages (`baseten`, `baseten-bdn`,
`baseten-loops`) are exempt, so the cookbook can move to their releases on the day they
ship. Add a new Baseten dependency to that list. When the newest fail, an upstream
release breaks the cookbook for users who install it: fix the code, or lower the cap.
When the floors fail, raise the floor. To use another package's release before it is 7
days old, add it to `exclude-newer-package` with a cutoff date, and a note to delete the
entry once the 7 days pass.

The **release-title** check keeps a PR into `main` from merging unless it comes from
`develop` or `hotfix/` and its title names a new version above the last release.

To fix a release while `develop` holds work that isn't ready to ship:

1. Branch `hotfix/X.Y.Z` from `main`, fix the bug, and open a PR into `main` titled
   "Release vX.Y.Z". Merging it releases as above.
2. PR `main` into `develop`, so the fix reaches it.

A pre-release is a pushed tag: `git tag v0.2.0-alpha.1 && git push origin
v0.2.0-alpha.1`. The tag must name a commit on `main` or `develop`, because a PyPI upload
is permanent. It builds as version `0.2.0a1` and is published as a pre-release. A pushed `vX.Y.Z` tag is refused; releases come only from a merged PR.
