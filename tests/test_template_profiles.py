from types import SimpleNamespace

import pytest

from pbrainz.api.control_routes import (
    activate_ui_template_profile,
    delete_ui_template_profile,
    reset_ui_template_profile,
    save_ui_template_profile,
)
from pbrainz.api.models import (
    UITemplateProfile,
    UITemplateProfileActionRequest,
    UITemplateProfileSaveRequest,
)
from pbrainz.config import Settings
from pbrainz.context_builder import ContextBuilder, ContextInput
from pbrainz.database import SettingsDatabase
from pbrainz.template_profiles import (
    DEFAULT_TEMPLATE_PROFILE_ID,
    active_template_profile,
    load_template_profiles,
    normalize_profile,
    render_template,
    template_profile_for_provider,
)


class FakeRegistry:
    provider_names = ("custom",)


def _request(tmp_path):
    settings = Settings(
        database_path=str(tmp_path / "settings.db"),
        enabled_providers="custom",
        custom_base_url="http://mock-provider",
        custom_models="fake-model",
    )
    database = SettingsDatabase(settings.database_path)
    database.initialize()
    app = SimpleNamespace(
        title="PBrainZ",
        state=SimpleNamespace(
            settings=settings,
            providers=FakeRegistry(),
            database=database,
            bridge_controller=SimpleNamespace(
                as_dict=lambda: {
                    "bridge": {
                        "available": False,
                        "enabled": False,
                        "ready": False,
                        "message": "test",
                    },
                    "worker_enabled": False,
                    "worker_running": False,
                }
            ),
            game_bridge_settings=SimpleNamespace(
                read=lambda: SimpleNamespace(enabled=False)
            ),
        ),
    )
    return SimpleNamespace(app=app)


def test_template_profile_normalization_keeps_identity_and_player_tokens() -> None:
    profile = normalize_profile(
        {
            "id": "My Horde Template",
            "name": "Horde test",
            "mode": "instruct",
            "context_template": "Say hello.",
        }
    )

    assert profile.id == "my-horde-template"
    assert "{{system}}" in profile.context_template
    assert "{{user}}" in profile.context_template
    assert render_template(
        "{{char}} -> {{user}} -> {{unknown}}",
        {"char": "Emilio", "user": "Hi"},
    ) == "Emilio -> Hi ->"


def test_instruct_profile_is_applied_as_one_rendered_prompt() -> None:
    profile = normalize_profile(
        {
            "id": "instruct",
            "name": "Instruct",
            "mode": "instruct",
            "system_prompt": "Always mention the weather.",
            "context_template": (
                "{{system}}\n{{history}}\n{{user_prefix}}{{user}}\n"
                "{{assistant_prefix}}"
            ),
        }
    )
    result = ContextBuilder(template_profile=profile).build(
        ContextInput(
            npc_name="Emilio",
            player_name="Alex",
            current_message="What is your name?",
        )
    )

    assert len(result.messages) == 1
    assert result.messages[0].role == "user"
    assert "Always mention the weather." in (result.messages[0].content or "")
    assert "User: What is your name?" in (result.messages[0].content or "")
    assert result.diagnostics["template_profile_mode"] == "instruct"


def test_horde_uses_instruct_only_for_the_untouched_native_default() -> None:
    assert (
        template_profile_for_provider(None, DEFAULT_TEMPLATE_PROFILE_ID, "horde").id
        == "instruct-text"
    )
    assert (
        template_profile_for_provider(None, DEFAULT_TEMPLATE_PROFILE_ID, "gemini").id
        == DEFAULT_TEMPLATE_PROFILE_ID
    )
    assert (
        template_profile_for_provider(None, "instruct-text", "gemini").id
        == "instruct-text"
    )
    assert (
        template_profile_for_provider(
            '{"profiles":[{"id":"custom-profile","name":"Custom","mode":"chat"}]}',
            "custom-profile",
            "horde",
        ).id
        == "custom-profile"
    )


@pytest.mark.asyncio
async def test_template_profiles_persist_activate_reset_and_delete(tmp_path) -> None:
    request = _request(tmp_path)
    profile = UITemplateProfile(
        id="custom-test",
        name="Custom test",
        mode="instruct",
        system_prompt="Stay concise.",
        context_template="{{system}}\n{{user_prefix}}{{user}}\n{{assistant_prefix}}",
        stop_sequences=["\nUser: "],
    )

    saved = await save_ui_template_profile(
        request,
        UITemplateProfileSaveRequest(profile=profile),
    )
    assert {item.id for item in saved.template_profiles} >= {
        DEFAULT_TEMPLATE_PROFILE_ID,
        "custom-test",
    }
    assert "custom-test" in request.app.state.database.load_settings()["template_profiles_json"]

    activated = await activate_ui_template_profile(
        request,
        UITemplateProfileActionRequest(profile_id="custom-test"),
    )
    assert activated.active_template_profile_id == "custom-test"
    assert request.app.state.settings.active_template_profile_id == "custom-test"

    reset = await reset_ui_template_profile(
        request,
        UITemplateProfileActionRequest(profile_id=DEFAULT_TEMPLATE_PROFILE_ID),
    )
    assert reset.active_template_profile_id == "custom-test"

    deleted = await delete_ui_template_profile(
        request,
        UITemplateProfileActionRequest(profile_id="custom-test"),
    )
    assert deleted.active_template_profile_id == DEFAULT_TEMPLATE_PROFILE_ID
    assert "custom-test" not in {item.id for item in deleted.template_profiles}


def test_corrupt_template_json_falls_back_to_builtin_profiles() -> None:
    profiles = load_template_profiles("not-json")
    assert active_template_profile("not-json", DEFAULT_TEMPLATE_PROFILE_ID).id == profiles[0].id
