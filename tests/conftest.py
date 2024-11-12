import pytest
from openai import OpenAI

@pytest.fixture
def openai_client():
    return OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )

@pytest.fixture
def character_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "characters",
            "schema": {
                "type": "object",
                "properties": {
                    "characters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "occupation": {"type": "string"},
                                "personality": {"type": "string"},
                                "background": {"type": "string"}
                            },
                            "required": ["name", "occupation", "personality", "background"]
                        },
                        "minItems": 1,
                    }
                },
                "required": ["characters"]
            },
        }
    } 