#!/usr/bin/env python3
"""Build a Windows executable or Linux AppImage for a GitHub release.

PyInstaller builds must run on the target operating system. AppImage creation
also requires a Linux host; appimagetool is downloaded automatically when it
is not already available on PATH.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = PROJECT_ROOT / "packaging"
BUILD_ROOT = PROJECT_ROOT / "build" / "release"
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "release"
PRODUCT_BINARY_NAME = "PBrainZ"
VERSION_FILE = PROJECT_ROOT / "src" / "pbrainz" / "version.py"
APPIMAGE_TOOL_URL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/"
    "appimagetool-{architecture}.AppImage"
)


def main() -> int:
    args = _parse_args()
    target = _resolve_target(args.target)
    output_dir = Path(args.output_dir).expanduser().resolve()
    configured_appimagetool = (
        Path(args.appimagetool).expanduser().resolve() if args.appimagetool else None
    )
    staging = BUILD_ROOT / target
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_release_output_is_safe(output_dir)
    version, previous_version = _prepare_build_version(args.version, args.bump)

    try:
        icon_assets = _prepare_icon_assets(staging)
        bundle = _build_pyinstaller(target, staging, icon_assets)
        if target == "exe":
            artifact = output_dir / f"{PRODUCT_BINARY_NAME}-{version}-{_platform_tag()}.exe"
            shutil.copy2(bundle, artifact)
        else:
            artifact = _build_appimage(
                bundle, staging, output_dir, version, configured_appimagetool, icon_assets
            )
        print(f"Release artifact: {artifact}")
        return 0
    except BaseException:
        if previous_version is not None:
            _write_version(previous_version)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("auto", "exe", "appimage"),
        default="auto",
        help="Artifact type; auto selects exe on Windows and AppImage on Linux.",
    )
    parser.add_argument(
        "--version",
        help="Explicit version to package; overrides automatic bumping (for example, v0.2.0).",
    )
    parser.add_argument(
        "--bump",
        choices=("patch", "minor", "major", "none"),
        default="patch",
        help=(
            "Automatic version bump for local builds; default: patch. "
            "Use none to rebuild unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory where the final artifact is written.",
    )
    parser.add_argument(
        "--appimagetool",
        help="Path to appimagetool; otherwise PATH or the official continuous build is used.",
    )
    return parser.parse_args()


def _resolve_target(requested: str) -> str:
    if requested != "auto":
        target = requested
    elif os.name == "nt":
        target = "exe"
    elif sys.platform.startswith("linux"):
        target = "appimage"
    else:
        raise SystemExit("Automatic release builds support Windows and Linux hosts only.")
    if target == "exe" and os.name != "nt":
        raise SystemExit("Build the Windows executable on a Windows host.")
    if target == "appimage" and not sys.platform.startswith("linux"):
        raise SystemExit("Build the AppImage on a Linux host.")
    return target


_VERSION_ASSIGNMENT_RE = re.compile(
    r"(?m)^(?P<prefix>\s*__version__\s*=\s*[\"'])(?P<version>[^\"']+)(?P<suffix>[\"']\s*)$"
)
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?P<suffix>[-+][0-9A-Za-z.-]+)?$"
)


def _read_version() -> str:
    match = _VERSION_ASSIGNMENT_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"Could not find __version__ in {VERSION_FILE}")
    return match.group("version")


def _validate_version(value: str) -> str:
    normalized = value.strip().removeprefix("v")
    if _SEMVER_RE.fullmatch(normalized) is None:
        raise SystemExit(f"Version must be semantic MAJOR.MINOR.PATCH, got: {value!r}")
    return normalized


def _bump_version(current: str, bump: str) -> str:
    match = _SEMVER_RE.fullmatch(_validate_version(current))
    assert match is not None
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    elif bump == "patch":
        patch += 1
    elif bump != "none":
        raise SystemExit(f"Unknown version bump: {bump!r}")
    return f"{major}.{minor}.{patch}"


def _write_version(version: str) -> None:
    version = _validate_version(version)
    source = VERSION_FILE.read_text(encoding="utf-8")
    updated, count = _VERSION_ASSIGNMENT_RE.subn(
        lambda match: f'{match.group("prefix")}{version}{match.group("suffix")}',
        source,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"Could not update __version__ in {VERSION_FILE}")
    VERSION_FILE.write_text(updated, encoding="utf-8")


def _release_version(requested: str | None, bump: str = "none") -> str:
    """Return the normalized version that a build should package."""
    return _validate_version(requested) if requested else _bump_version(_read_version(), bump)


def _prepare_build_version(
    requested: str | None,
    bump: str,
) -> tuple[str, str | None]:
    """Set the package version for this build and return its rollback value."""
    current = _read_version()
    version = _release_version(requested, bump)
    if version == current:
        return version, None
    _write_version(version)
    print(f"Version: {current} -> {version}")
    return version, current


def _prepare_icon_assets(staging: Path) -> Path:
    """Copy the checked-in raster variants before freezing the app.

    The release runners do not share native graphics libraries. In particular,
    CairoSVG needs a separately installed Cairo runtime on Windows, while the
    repository already contains the canonical PNG variants used by the GUI.
    Reusing those files keeps release builds deterministic and platform-neutral.
    """

    icon_assets = staging / "icon-assets"
    icon_assets.mkdir(parents=True, exist_ok=True)
    source = PACKAGING_ROOT / "pbrainz.svg"
    windows_icon = PACKAGING_ROOT / "pbrainz.ico"
    gui_assets = PROJECT_ROOT / "src" / "pbrainz" / "gui" / "assets"
    for filename in ("pbrainz.png", "pbrainz-mark.png"):
        asset = gui_assets / filename
        if not asset.is_file():
            raise SystemExit(f"Expected checked-in icon asset was not found: {asset}")
        shutil.copy2(asset, icon_assets / filename)
    if not windows_icon.is_file():
        raise SystemExit(f"Expected checked-in Windows icon asset was not found: {windows_icon}")
    shutil.copy2(windows_icon, icon_assets / windows_icon.name)
    shutil.copy2(source, icon_assets / "pbrainz.svg")
    return icon_assets


def _build_pyinstaller(target: str, staging: Path, icon_assets: Path) -> Path:
    _run([sys.executable, "-m", "PyInstaller", "--version"])
    dist_path = staging / "pyinstaller-dist"
    work_path = staging / "pyinstaller-work"
    spec_path = staging / "pyinstaller-spec"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        # Release artifacts are GUI applications. In particular, the Windows
        # build must use the windowed subsystem so launching it from Explorer
        # does not create a second command prompt window.
        "--windowed",
        "--name",
        PRODUCT_BINARY_NAME,
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--add-data",
        f"{icon_assets / 'pbrainz.png'}{os.pathsep}pbrainz/gui/assets",
        "--add-data",
        f"{icon_assets / 'pbrainz-mark.png'}{os.pathsep}pbrainz/gui/assets",
        "--add-data",
        f"{icon_assets / 'pbrainz.svg'}{os.pathsep}pbrainz/gui/assets",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        "--hidden-import",
        "google.genai",
        "--hidden-import",
        "google.genai.types",
        "--hidden-import",
        "openai",
        "--hidden-import",
        "piper",
        "--collect-submodules",
        "uvicorn",
        "--collect-data",
        "google.genai",
        "--collect-data",
        "piper",
    ]
    if target == "exe":
        command.extend(["--icon", str(icon_assets / "pbrainz.ico")])
    command.append("--onefile" if target == "exe" else "--onedir")
    command.append(str(PACKAGING_ROOT / "launcher.py"))
    _run(command)
    if target == "exe":
        executable = dist_path / f"{PRODUCT_BINARY_NAME}.exe"
    else:
        executable = dist_path / PRODUCT_BINARY_NAME / PRODUCT_BINARY_NAME
    if not executable.exists():
        raise SystemExit(f"PyInstaller completed but expected output was not found: {executable}")
    return executable


def _ensure_release_output_is_safe(output_dir: Path) -> None:
    """Reject output folders that already contain private runtime data.

    PBrainZ creates ``data/`` beside a portable executable at first launch.
    That directory contains credentials and conversation history, so a release
    build must never reuse an output folder that already has it. The data is
    intentionally not created by this build script.
    """
    runtime_data = output_dir / "data"
    if runtime_data.exists():
        raise SystemExit(
            "Refusing to build a release beside private runtime data: "
            f"{runtime_data}. Choose a clean --output-dir or move that data "
            "directory before packaging."
        )


def _build_appimage(
    bundle: Path,
    staging: Path,
    output_dir: Path,
    version: str,
    configured_tool: Path | None,
    icon_assets: Path,
) -> Path:
    app_dir = staging / "AppDir"
    payload = app_dir / "usr" / "lib" / PRODUCT_BINARY_NAME
    payload.parent.mkdir(parents=True)
    shutil.copytree(bundle.parent, payload)
    shutil.copy2(PACKAGING_ROOT / "pbrainz.desktop", app_dir / "pbrainz.desktop")
    shutil.copy2(icon_assets / "pbrainz.svg", app_dir / "pbrainz.svg")
    shutil.copy2(icon_assets / "pbrainz.png", app_dir / "pbrainz.png")
    icon_dir = app_dir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps"
    icon_dir.mkdir(parents=True)
    shutil.copy2(icon_assets / "pbrainz.svg", icon_dir / "pbrainz.svg")
    for size in (128, 256):
        png_icon_dir = (
            app_dir
            / "usr"
            / "share"
            / "icons"
            / "hicolor"
            / f"{size}x{size}"
            / "apps"
        )
        png_icon_dir.mkdir(parents=True)
        shutil.copy2(icon_assets / "pbrainz.png", png_icon_dir / "pbrainz.png")
    app_run = app_dir / "AppRun"
    app_run.write_text(
        '#!/bin/sh\n'
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        'if [ -z "${PBRAINZ_PORTABLE_ROOT:-}" ]; then\n'
        '    if [ -n "${APPIMAGE:-}" ]; then\n'
        '        export PBRAINZ_PORTABLE_ROOT="$(dirname "$(readlink -f "$APPIMAGE")")"\n'
        '    else\n'
        '        export PBRAINZ_PORTABLE_ROOT="$HERE"\n'
        '    fi\n'
        'fi\n'
        'exec "$HERE/usr/lib/PBrainZ/PBrainZ" "$@"\n',
        encoding="utf-8",
    )
    app_run.chmod(app_run.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    tool = _find_appimagetool(configured_tool, staging)
    artifact = output_dir / f"{PRODUCT_BINARY_NAME}-{version}-{_platform_tag()}.AppImage"
    temporary_artifact = output_dir / f".{artifact.name}.tmp"
    temporary_artifact.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    try:
        _run([tool, str(app_dir), str(temporary_artifact)], environment=environment)
        if not temporary_artifact.exists():
            raise SystemExit(
                f"appimagetool completed but expected output was not found: "
                f"{temporary_artifact}"
            )
        # Rename only after appimagetool has finished writing. This also lets
        # Linux replace an older AppImage that is still running.
        os.replace(temporary_artifact, artifact)
    finally:
        temporary_artifact.unlink(missing_ok=True)
    artifact.chmod(artifact.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return artifact


def _find_appimagetool(configured_tool: Path | None, staging: Path) -> str:
    if configured_tool:
        if not configured_tool.is_file():
            raise SystemExit(f"The configured appimagetool was not found: {configured_tool}")
        return str(Path(configured_tool).expanduser().resolve())
    available = shutil.which("appimagetool")
    if available:
        return available
    architecture = _appimage_architecture()
    destination = staging / "appimagetool.AppImage"
    url = APPIMAGE_TOOL_URL.format(architecture=architecture)
    print(f"Downloading appimagetool for {architecture}...")
    urllib.request.urlretrieve(url, destination)
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(destination)


def _appimage_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    raise SystemExit(f"No automatic appimagetool download is configured for {machine}.")


def _platform_tag() -> str:
    machine = platform.machine().lower()
    normalized = {"amd64": "x86_64", "x86-64": "x86_64", "arm64": "aarch64"}
    return normalized.get(machine, machine or "unknown")


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)
    except FileNotFoundError as error:
        raise SystemExit(f"Required build command is unavailable: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Release build command failed with exit code {error.returncode}."
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
