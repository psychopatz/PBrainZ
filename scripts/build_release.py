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
import shutil
import stat
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = PROJECT_ROOT / "packaging"
BUILD_ROOT = PROJECT_ROOT / "build" / "release"
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "release"
PRODUCT_BINARY_NAME = "PBrainZ"
APPIMAGE_TOOL_URL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/"
    "appimagetool-{architecture}.AppImage"
)


def main() -> int:
    args = _parse_args()
    target = _resolve_target(args.target)
    version = _release_version(args.version)
    output_dir = Path(args.output_dir).expanduser().resolve()
    configured_appimagetool = (
        Path(args.appimagetool).expanduser().resolve() if args.appimagetool else None
    )
    staging = BUILD_ROOT / target
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("auto", "exe", "appimage"),
        default="auto",
        help="Artifact type; auto selects exe on Windows and AppImage on Linux.",
    )
    parser.add_argument("--version", help="Release version used in the artifact filename.")
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


def _release_version(requested: str | None) -> str:
    if requested:
        value = requested.removeprefix("v")
    else:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            value = tomllib.load(handle)["project"]["version"]
    return "".join(
        character if character.isalnum() or character in ".-_" else "-" for character in value
    )


def _prepare_icon_assets(staging: Path) -> Path:
    """Render all PNG variants from the canonical SVG before freezing the app."""

    icon_assets = staging / "icon-assets"
    source = PACKAGING_ROOT / "pbrainz.svg"
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "render_icon.py"),
            "--source",
            str(source),
            "--output",
            str(icon_assets / "pbrainz.png"),
            "--size",
            "256",
        ]
    )
    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "render_icon.py"),
            "--source",
            str(source),
            "--output",
            str(icon_assets / "pbrainz-mark.png"),
            "--size",
            "512",
            "--transparent",
        ]
    )
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
        # Keep stdout/stderr available so ``PBrainZ --activity`` and the
        # bridge send/receive diagnostics work from a terminal in a frozen
        # release build. The GUI still launches normally from the desktop.
        "--console",
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
        'if [ -n "${APPIMAGE:-}" ]; then\n'
        '    export PBRAINZ_PORTABLE_ROOT="$(dirname "$(readlink -f "$APPIMAGE")")"\n'
        'else\n'
        '    export PBRAINZ_PORTABLE_ROOT="$HERE"\n'
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
