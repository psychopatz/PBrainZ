import importlib.util
from pathlib import Path

import pytest

from pbrainz import __version__
from pbrainz.branding import PRODUCT_BINARY_NAME, PRODUCT_VERSION

_BUILD_RELEASE_PATH = Path(__file__).parents[1] / "scripts" / "build_release.py"
_BUILD_RELEASE_SPEC = importlib.util.spec_from_file_location(
    "pbrainz_build_release", _BUILD_RELEASE_PATH
)
assert _BUILD_RELEASE_SPEC and _BUILD_RELEASE_SPEC.loader
build_release = importlib.util.module_from_spec(_BUILD_RELEASE_SPEC)
_BUILD_RELEASE_SPEC.loader.exec_module(build_release)


def test_package_and_branding_use_the_same_pre_one_version() -> None:
    assert __version__ == PRODUCT_VERSION
    assert __version__.split(".")[0] == "0"


def test_window_branding_uses_binary_name_and_current_version() -> None:
    assert f"{PRODUCT_BINARY_NAME} v{PRODUCT_VERSION}" == "PBrainZ v" + __version__


@pytest.mark.parametrize(
    ("current", "bump", "expected"),
    [
        ("0.1.0", "patch", "0.1.1"),
        ("0.1.9", "minor", "0.2.0"),
        ("0.9.9", "major", "1.0.0"),
        ("1.2.3", "none", "1.2.3"),
    ],
)
def test_version_bumps_are_semantic(current: str, bump: str, expected: str) -> None:
    assert build_release._bump_version(current, bump) == expected


def test_requested_version_normalizes_tag_prefix() -> None:
    assert build_release._release_version("v0.2.0") == "0.2.0"


def test_prepare_build_version_updates_only_the_version_source(tmp_path, monkeypatch) -> None:
    version_file = tmp_path / "version.py"
    version_file.write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr(build_release, "VERSION_FILE", version_file)

    version, previous = build_release._prepare_build_version(None, "patch")

    assert version == "0.1.1"
    assert previous == "0.1.0"
    assert version_file.read_text(encoding="utf-8") == '__version__ = "0.1.1"\n'


def test_invalid_version_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_release._release_version("version-one")


def test_version_source_is_inside_the_package() -> None:
    assert Path(build_release.VERSION_FILE).name == "version.py"
    assert Path(build_release.VERSION_FILE).exists()


def test_release_icon_assets_include_native_windows_icon(tmp_path) -> None:
    assets = build_release._prepare_icon_assets(tmp_path)

    assert (assets / "pbrainz.ico").is_file()
    assert (assets / "pbrainz.png").is_file()
    assert (assets / "pbrainz-mark.png").is_file()
    assert (assets / "pbrainz.svg").is_file()


def test_windows_pyinstaller_build_is_windowed_and_uses_pbrainz_icon(tmp_path, monkeypatch) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    icon_assets = build_release._prepare_icon_assets(staging)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        if len(commands) == 2:
            executable = staging / "pyinstaller-dist" / "PBrainZ.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"test executable")

    monkeypatch.setattr(build_release, "_run", fake_run)

    executable = build_release._build_pyinstaller("exe", staging, icon_assets)

    assert executable.name == "PBrainZ.exe"
    pyinstaller_command = commands[1]
    assert "--windowed" in pyinstaller_command
    assert "--console" not in pyinstaller_command
    icon_index = pyinstaller_command.index("--icon")
    assert pyinstaller_command[icon_index + 1] == str(icon_assets / "pbrainz.ico")


def test_release_builder_rejects_private_runtime_data(tmp_path) -> None:
    (tmp_path / "data").mkdir()

    with pytest.raises(SystemExit, match="private runtime data"):
        build_release._ensure_release_output_is_safe(tmp_path)
