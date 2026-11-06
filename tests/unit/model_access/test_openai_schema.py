"""The generated Chat Completions types come from one pinned OpenAI revision."""

import pytest
from pydantic import ValidationError
from support.unit_rest import load_script

from loops_cookbook.model_access.openai.schema.models import CreateChatCompletionRequest


@pytest.fixture(scope="module")
def generator():
    return load_script("generate_openai_models")


def test_the_source_is_the_pinned_revision(generator):
    assert generator.REVISION in generator.SOURCE_URL


def test_the_extract_holds_exactly_the_schemas_the_roots_reach(generator):
    document = {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "1"},
        "components": {
            "schemas": {
                "CreateChatCompletionRequest": {"$ref": "#/components/schemas/Shared"},
                "CreateChatCompletionResponse": {"items": [{"$ref": "#/components/schemas/B"}]},
                "CreateChatCompletionStreamResponse": {"type": "object"},
                "Shared": {"properties": {"b": {"$ref": "#/components/schemas/B"}}},
                "B": {"type": "string"},
                "Unrelated": {"type": "integer"},
            }
        },
    }
    extracted = generator.extract(document)["components"]["schemas"]
    assert sorted(extracted) == sorted([*generator.ROOTS, "Shared", "B"])


def test_the_check_fails_when_the_models_were_edited(generator, tmp_path, monkeypatch):
    edited = tmp_path / "models.py"
    edited.write_text(generator.MODELS.read_text() + "# edited by hand\n")
    monkeypatch.setattr(generator, "MODELS", edited)
    assert generator.check() == ["models.py differs from what the generator produces"]


def test_unknown_fields_are_kept_for_the_policy_to_reject():
    request = CreateChatCompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "not_a_field": 1}
    )
    assert request.model_extra == {"not_a_field": 1}


def test_malformed_values_fail_schema_validation():
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate({"model": "m", "messages": []})
    with pytest.raises(ValidationError):
        CreateChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 3}
        )
