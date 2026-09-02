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
