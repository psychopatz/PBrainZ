from pbrainz.providers.gemini import GeminiProvider


def test_gemini_schema_replaces_legacy_bridge_depth_placeholders() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "[depth-limit]"},
            "intensity": {"type": "[unsupported]"},
        },
        "additionalProperties": False,
    }

    sanitized = GeminiProvider._sanitize_schema(schema)

    assert sanitized == {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "intensity": {"type": "string"},
        },
    }
