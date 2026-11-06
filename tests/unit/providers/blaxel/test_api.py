"""The Blaxel facade's image builds: what a specification describes and what is built."""

from __future__ import annotations

import asyncio
import json
import tomllib

import httpx
import pytest

blaxel = pytest.importorskip("blaxel.core")

from loops_cookbook.providers.blaxel import api
from loops_cookbook.providers.blaxel.api import (
    BlaxelApi,
    image_build_specification,
    image_context,
)
from loops_cookbook.providers.blaxel.build_file import build_toml
from loops_cookbook.providers.blaxel.incidents import creation_disposition, status_code
from loops_cookbook.resources.api import CreationDisposition


@pytest.fixture
def context(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("hello")
    return blaxel.ImageBuildContext(
        base_image="python:3.12-slim",
        instructions=["RUN pip install pytest", "WORKDIR /app"],
        local_files=[
            blaxel.LocalFile(
                source_path=source.resolve(),
                destination_path="/app/data.txt",
                context_name="data.txt",
            )
        ],
        has_entrypoint=True,
    )


@pytest.mark.parametrize("files", [True, False], ids=["with_files", "without_files"])
def test_a_build_specification_is_journal_safe_and_describes_its_image_exactly(context, files):
    if not files:
        context.local_files = []
    specification = image_build_specification(
        blaxel.ImageInstance(context), memory=4096, sandbox_version="v2"
    )
    reread = json.loads(json.dumps(specification))  # what the journal gives back
    assert image_context(reread) == context
    assert (reread["memory"], reread["sandbox_version"]) == (4096, "v2")


@pytest.fixture
def built(monkeypatch):
    """What each image build packaged: its context files, read while it built."""
    packaged = []

    async def build(image, *, name, memory, timeout, sandbox_version, **build_sizes):
        packaged.append(
            {item.context_name: item.source_path.read_text() for item in image._context.local_files}
        )
        packaged.append([item.source_path for item in image._context.local_files])

    async def update_ttl(name, ttl):
        pass

    monkeypatch.setattr(api._LabeledImage, "build", build)
    monkeypatch.setattr(api.SandboxInstance, "update_ttl", update_ttl)
    return packaged


REGION = "us-was-1"


def specification(context, **options):
    return json.loads(
        json.dumps(
            image_build_specification(
                blaxel.ImageInstance(context), memory=4096, sandbox_version="v2", **options
            )
        )
    )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"slim": False}, build_toml("us-was-1", slim=False)),
        ({}, build_toml("us-was-1", slim=True)),  # Blaxel's own slimming, still its region
    ],
    ids=["unslimmed", "default"],
)
async def test_every_build_packages_a_blaxel_toml_naming_the_region_only_while_it_builds(
    context, built, options, expected
):
    await BlaxelApi().build_image("img", specification(context, **options), {}, 10, region=REGION)
    contents, paths = built
    assert contents == {"data.txt": "hello", "blaxel.toml": expected}
    assert all(path.exists() for path in paths if path.name == "data.txt")
    assert not any(path.exists() for path in paths if path.name == "blaxel.toml")


async def test_a_tasks_own_blaxel_toml_is_kept(context, built, tmp_path):
    own = tmp_path / "blaxel.toml"
    own.write_text("[build]\nslim = true\n")
    context.local_files.append(
        blaxel.LocalFile(
            source_path=own, destination_path="blaxel.toml", context_name="blaxel.toml"
        )
    )
    await BlaxelApi().build_image("img", specification(context, slim=False), {}, 10, region=REGION)
    assert tomllib.loads(built[0]["blaxel.toml"]) == {"build": {"slim": True, "region": REGION}}
    assert own.read_text() == "[build]\nslim = true\n"  # the task's file is never written


@pytest.mark.parametrize(
    "text",
    ['[build]\nregion = "us-was-1"\nslim = true\n', '[entrypoint]\nprod = "x"\n'],
    ids=["the same region", "no [build] table"],
)
async def test_a_tasks_blaxel_toml_keeps_its_keys_and_builds_in_the_region(
    context, built, tmp_path, text
):
    own = tmp_path / "blaxel.toml"
    own.write_text(text)
    context.local_files.append(
        blaxel.LocalFile(
            source_path=own, destination_path="blaxel.toml", context_name="blaxel.toml"
        )
    )
    await BlaxelApi().build_image("img", specification(context), {}, 10, region=REGION)
    shipped = tomllib.loads(built[0]["blaxel.toml"])
    original = tomllib.loads(text)
    assert shipped == {**original, "build": {**original.get("build", {}), "region": REGION}}


async def test_a_tasks_blaxel_toml_naming_another_region_is_refused(context, built, tmp_path):
    own = tmp_path / "blaxel.toml"
    own.write_text('[build]\nregion = "eu-lon-1"\n')
    context.local_files.append(
        blaxel.LocalFile(
            source_path=own, destination_path="blaxel.toml", context_name="blaxel.toml"
        )
    )
    with pytest.raises(ValueError, match="builds in region 'eu-lon-1'"):
        await BlaxelApi().build_image("img", specification(context), {}, 10, region=REGION)
    assert built == []  # nothing was sent to build


async def test_the_sdks_build_creates_a_builder_with_our_labels_and_its_own_in_the_region(
    context, monkeypatch
):
    """Through the SDK's own ``build``, which hands its build settings to the payload."""
    sent = []

    async def create(image, sandbox):
        sent.append(sandbox)
        raise RuntimeError("stop before any upload")

    monkeypatch.setattr(api._LabeledImage, "_create_sandbox_with_upload", create)
    image = api._LabeledImage(context, {"loops.run": "r"}, REGION)
    with pytest.raises(RuntimeError, match="stop before any upload"):
        await image.build(name="b", timeout=1)  # as build_image calls it, region via blaxel.toml
    [payload] = sent
    labels = payload.metadata.labels
    assert payload.spec.region == REGION and labels["loops.run"] == "r"
    assert labels["x-blaxel-auto-generated"] == "true"  # the SDK's own label is kept


@pytest.mark.parametrize(
    ("sizes", "labels"),
    [
        ({}, {}),
        (
            {"build_memory": 16384, "build_volume": 65536},
            {"x-blaxel-build-memory": "16384", "x-blaxel-build-volume": "65536"},
        ),
    ],
    ids=["blaxels_defaults", "sized"],
)
async def test_a_build_asks_blaxel_for_the_build_memory_and_disk_it_is_given(
    context, monkeypatch, sizes, labels
):
    sent = []

    async def create(image, sandbox):
        sent.append(sandbox)
        raise RuntimeError("stop before any upload")

    async def update_ttl(name, ttl):
        pass  # build_image's cleanup: never reach Blaxel from a unit test

    monkeypatch.setattr(api._LabeledImage, "_create_sandbox_with_upload", create)
    monkeypatch.setattr(api.SandboxInstance, "update_ttl", update_ttl)
    with pytest.raises(RuntimeError, match="stop before any upload"):
        await BlaxelApi().build_image("img", specification(context), {}, 10, region=REGION, **sizes)
    [payload] = sent
    asked = {
        k: v
        for k, v in payload.metadata.labels.to_dict().items()
        if k.startswith("x-blaxel-build-")
    }
    assert asked == labels


async def test_the_workspace_is_listed_page_by_page_to_its_last(monkeypatch):
    """Each SDK page is yielded as it arrives, so a caller can bound each one."""
    from types import SimpleNamespace as NS

    def sandbox(name):
        return NS(metadata=NS(to_dict=lambda: {"name": name, "labels": {}}), status="DEPLOYED")

    class Page:
        def __init__(self, names, rest):
            self.data, self._rest = [sandbox(n) for n in names], rest
            self.has_more = bool(rest)

        async def next_page(self):
            return Page(self._rest[0], self._rest[1:])

    async def first(limit):
        return Page(["a", "b"], [["c", "d"], ["e"]])

    monkeypatch.setattr(api.SandboxInstance, "list", first)
    pages = [[observed.name for observed in page] async for page in BlaxelApi().list()]
    assert pages == [["a", "b"], ["c", "d"], ["e"]]


# What Blaxel answered a burst of creates on 2026-10-01: blaxel-probe/FACTS.md.
RATE_LIMITED = (
    b'{"code":"WORKSPACE_RATE_LIMIT_EXCEEDED","message":"Too many sandboxes are being '
    b"created concurrently for workspace 'team'. Please try again in a few moments.\","
    b'"status_code":429}'
)


@pytest.fixture
async def control_plane(monkeypatch):
    """The SDK's own client, answering a builder's create with a rate limit."""
    seen = []

    def answer(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/sandboxes":
            return httpx.Response(429, content=RATE_LIMITED)
        return httpx.Response(404, json={"code": 404, "error": "not found"})

    mocked = httpx.AsyncClient(
        base_url="https://api.blaxel.test", transport=httpx.MockTransport(answer)
    )
    # The SDK keeps one client per event loop; this loop's is the mock.
    monkeypatch.setattr(api.client, "_async_client", mocked)
    monkeypatch.setattr(api.client, "_async_client_loop", asyncio.get_running_loop())
    yield seen
    await mocked.aclose()


async def test_a_builder_create_blaxel_rate_limits_fails_with_its_status(context, control_plane):
    with pytest.raises(Exception) as failure:
        await BlaxelApi().build_image("img", specification(context), {}, 10, region=REGION)
    assert status_code(failure.value) == 429
    assert creation_disposition(failure.value) is CreationDisposition.DEFINITELY_NOT_CREATED
    assert ("POST", "/sandboxes") in control_plane


# ------------------------------------------------------------- whether an image is built

IMAGE_NOT_FOUND = {"code": 404, "error": "image not found"}


def described(status, tags):
    """What ``GET /images/sandbox/<name>`` answered for a built image on 2026-10-03."""
    return {
        "metadata": {"name": "img", "resourceType": "sandbox", "status": status},
        "spec": {"size": 1024, "tags": tags},
    }


TAG = {"name": "latest", "createdAt": "2026-10-03T00:00:00Z", "size": 1024}


def counted(status, tag_count):
    """What it answers since: a count of tags, no list (2026-10-22)."""
    return {
        "metadata": {"name": "img", "resourceType": "sandbox", "status": status},
        "spec": {"size": 346841088, "tagCount": tag_count},
    }


@pytest.fixture
async def images(monkeypatch):
    """The SDK's own client, answering each image lookup as the test sets it."""
    state = {"answer": httpx.Response(404, json=IMAGE_NOT_FOUND), "seen": []}

    def answer(request):
        state["seen"].append((request.method, request.url.path))
        if isinstance(state["answer"], Exception):
            raise state["answer"]
        return state["answer"]

    mocked = httpx.AsyncClient(
        base_url="https://api.blaxel.test", transport=httpx.MockTransport(answer)
    )
    monkeypatch.setattr(api.client, "_async_client", mocked)
    monkeypatch.setattr(api.client, "_async_client_loop", asyncio.get_running_loop())
    yield state
    await mocked.aclose()


@pytest.mark.parametrize(
    ("answer", "built"),
    [
        (httpx.Response(404, json=IMAGE_NOT_FOUND), False),
        (httpx.Response(200, json=described("BUILT", [TAG])), True),
        # A sandbox starts from :latest; after a failed or running rebuild it does not
        # start, although the earlier tag is still listed (3 Oct 2026).
        (httpx.Response(200, json=described("FAILED", [TAG])), False),
        (httpx.Response(200, json=described("BUILDING", [TAG])), False),
        (httpx.Response(200, json=described("BUILT", [])), False),
        (httpx.Response(200, json={"metadata": {"status": "BUILT"}}), False),
        (httpx.Response(200, json=counted("BUILT", 1)), True),
        (httpx.Response(200, json=counted("FAILED", 1)), False),
        (httpx.Response(200, json=counted("BUILT", 0)), False),
    ],
    ids=[
        "never_built",
        "built",
        "failed_with_tag",
        "building_with_tag",
        "no_tags",
        "no_spec",
        "built_counted",
        "failed_counted",
        "none_counted",
    ],
)
async def test_an_image_is_built_when_its_by_name_lookup_is_built_with_a_tag(images, answer, built):
    images["answer"] = answer
    assert await BlaxelApi().image_built("img") is built
    # One lookup by name, never the workspace's image list (502 on 19k images).
    assert images["seen"] == [("GET", "/images/sandbox/img")]


@pytest.mark.parametrize(
    ("answer", "raised"),
    [
        (httpx.Response(502, text="Bad Gateway"), httpx.HTTPStatusError),
        (httpx.Response(429, text="Too Many Requests"), httpx.HTTPStatusError),
        (httpx.Response(200, text="not json"), ValueError),
        (httpx.Response(200, json=[described("BUILT", [TAG])]), TypeError),
        (httpx.ConnectError("unreachable"), httpx.ConnectError),
    ],
    ids=["bad_gateway", "rate_limited", "bad_json", "not_an_object", "unreachable"],
)
async def test_an_image_lookup_without_an_answer_raises_rather_than_reads_as_unbuilt(
    images, answer, raised
):
    """Only a 404 proves an image unbuilt: a build in its place can break ``:latest``."""
    images["answer"] = answer
    with pytest.raises(raised):
        await BlaxelApi().image_built("img")


def tagged(*names, more=False):
    """What ``GET /images/sandbox/<name>/tags`` answered on 2026-11-06."""
    data = [{"name": name, "createdAt": "2026-11-06T06:46:30Z", "size": 1024} for name in names]
    return {"data": data, "meta": {"hasMore": more}}


@pytest.mark.parametrize(
    ("answer", "tags"),
    [
        (httpx.Response(404, json=IMAGE_NOT_FOUND), None),
        (httpx.Response(200, json=tagged("built")), frozenset({"built"})),
        (httpx.Response(200, json=tagged("v1", "v2")), frozenset({"v1", "v2"})),
        (httpx.Response(200, json=tagged()), frozenset()),
    ],
    ids=["no_image", "one_tag", "two_tags", "no_tags"],
)
async def test_an_images_tags_are_named_by_its_tags_lookup(images, answer, tags):
    images["answer"] = answer
    assert await BlaxelApi().image_tags("img") == tags
    assert images["seen"] == [("GET", "/images/sandbox/img/tags")]


@pytest.mark.parametrize(
    ("answer", "raised"),
    [
        (httpx.Response(502, text="Bad Gateway"), httpx.HTTPStatusError),
        (httpx.Response(200, json={"data": "nope"}), TypeError),
        (httpx.Response(200, json=tagged("built", more=True)), ValueError),
        (httpx.Response(200, json={"data": [{}], "meta": {}}), TypeError),
        (httpx.Response(200, json={"data": [{"name": 5}], "meta": {}}), TypeError),
        (httpx.Response(200, json={"data": ["built"], "meta": {}}), TypeError),
    ],
    ids=["bad_gateway", "not_a_list", "more_than_a_page", "no_name", "name_not_text", "not_a_tag"],
)
async def test_a_tags_lookup_without_a_whole_answer_raises(images, answer, raised):
    """A tag missing from a partial answer would read as absent: it raises instead."""
    images["answer"] = answer
    with pytest.raises(raised):
        await BlaxelApi().image_tags("img")
