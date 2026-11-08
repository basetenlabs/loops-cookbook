# Contributing

Open pull requests (PRs) against `develop`. CI must pass before a PR merges, and a code
owner from [.github/CODEOWNERS](.github/CODEOWNERS) must approve it.

## Commits and PR titles

`develop` takes squash merges, so a PR's title becomes its commit there. Write the
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
merge. PRs into `main` are releases, and they keep the `Release vX.Y.Z` title. See
[Releasing](#releasing).

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

Work lands on `develop`, by squash merge. `main` holds only released code. Every PR into
`main` is a release, from `develop` or from a `hotfix/` branch. It merges with a merge
commit, never a squash or a rebase. That way `main` and `develop` keep one history, and
the next release PR carries only new commits. The version comes from the git tag. Only
the README's install lines name it.

To release what is on `develop`:

1. If the release moves the `blaxel` range in `pyproject.toml`, run the live Blaxel
   tests that [tests/README.md](tests/README.md) describes. Pyright and the unit tests
   can't show how the SDK behaves against the real service. Daytona has no live tests
   yet, so check a move of the `daytona` range by hand.
2. On `develop`, change the tag in the README's `uv add` lines to the new version.
3. Run the **prepare-release** workflow with the version, such as `0.2.0`. To run it, go
   to Actions → prepare-release → Run workflow. The workflow checks that the version is
   new and above the last release. Then it opens a PR from `develop` into `main`, titled
   "Release v0.2.0", with a preview of the release notes. To do the same by hand, run
   `gh pr create -B main -H develop -t "Release v0.2.0"`.
4. CI on a PR that the workflow opened waits for a maintainer. Start it with
   **Approve and run**.
5. Merge the PR with **Create a merge commit**. The push to `main` starts the
   **release** workflow, which finds the PR that made the merge commit and tags the
   commit with the version in its title. The PR's required checks already ran lint,
   pyright and the tests against `uv.lock` on the same tree, so the workflow doesn't
   run them again. It builds the wheel and the sdist, and checks that they carry the
   tag's version. At the same time it tests both ends of the dependency ranges, which
   is the **dependency-ranges** check.
6. Approve the **pypi** deployment. The upload waits for the build and for both
   dependency-ranges checks. It publishes the wheel and the sdist to PyPI. Then the
   workflow publishes a GitHub release with them attached. The release also lists the
   dependency versions of both checks, and the notes since the last release.

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

The **release-title** check keeps a PR into `main` from merging unless the PR comes from
`develop` or a `hotfix/` branch, and its title names a new version above the last
release.

To fix a release while `develop` holds work that isn't ready to ship:

1. Branch `hotfix/X.Y.Z` from `main`, fix the bug, and open a PR into `main` titled
   "Release vX.Y.Z". Merging it releases the fix, as described above.
2. Open a PR from `main` into `develop`, so that the fix reaches `develop`.

To publish a pre-release, push a pre-release tag, such as with
`git tag v0.2.0-alpha.1 && git push origin v0.2.0-alpha.1`. The tag must name a commit
on `main` or `develop`, because an upload to PyPI is permanent. That tag builds as
version `0.2.0a1`, and it is published as a pre-release. A pre-release has no PR to run
the checks, so the workflow runs lint, pyright and the tests itself. A pushed `vX.Y.Z`
tag is refused, because releases come only from a merged PR.

## Diagrams

Most diagrams in [docs/assets](docs/assets/) have a source, the `.excalidraw` file of
the same name. To change a diagram, open its file at
[excalidraw.com](https://excalidraw.com), edit it and save it back. Then export it
twice, with "Embed scene" and the background turned on. Export it once in the light
theme, over `<name>.svg`, and once in the dark theme, over `<name>-dark.svg`.
