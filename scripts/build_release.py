#!/usr/bin/env python3
"""Build a Windows executable or Linux AppImage for a GitHub release.

PyInstaller builds must run on the target operating system. AppImage creation
also requires a Linux host; appimagetool is downloaded automatically when it
is not already available on PATH.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGING_ROOT = PROJECT_ROOT / "packaging"
BUILD_ROOT = PROJECT_ROOT / "build" / "release"
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "release"
PRODUCT_BINARY_NAME = "PBrainZ"
PROCESS_CLOSE_TIMEOUT_SECONDS = 8.0
VERSION_FILE = PROJECT_ROOT / "src" / "pbrainz" / "version.py"
APPIMAGE_TOOL_URL = (
    "https://github.com/AppImage/appimagetool/releases/download/continuous/"
    "appimagetool-{architecture}.AppImage"
)


def main() -> int:
    args = _parse_args()
    if args.keep_running and args.restart:
        raise SystemExit("--restart requires the default process shutdown; remove --keep-running.")
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
        if not args.keep_running:
            _close_running_instances()
        bundle = _build_pyinstaller(target, staging, icon_assets)
        if target == "exe":
            artifact = output_dir / f"{PRODUCT_BINARY_NAME}-{version}-{_platform_tag()}.exe"
            shutil.copy2(bundle, artifact)
        else:
            artifact = _build_appimage(
                bundle, staging, output_dir, version, configured_appimagetool, icon_assets
            )
        print(f"Release artifact: {artifact}")
        if args.restart:
            _restart_artifact(artifact)
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
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Do not stop an existing PBrainZ process before building.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Launch the newly built artifact after the build succeeds.",
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


def _close_running_instances(
    *, timeout: float = PROCESS_CLOSE_TIMEOUT_SECONDS
) -> tuple[int, ...]:
    """Gracefully stop only exact PBrainZ executable processes.

    Process matching deliberately ignores arbitrary command-line text, so a
    build path such as ``/Projects/PBrainZ`` cannot make the builder kill
    itself or an unrelated process.  Windows uses ``taskkill`` because it can
    close the executable tree; POSIX sends TERM first and KILL only after the
    bounded grace period.
    """
    pids = tuple(_running_pbrainz_pids())
    if not pids:
        return ()
    print(f"Stopping existing PBrainZ process(es): {', '.join(map(str, pids))}")
    if os.name == "nt":
        for pid in pids:
            _taskkill(pid, force=False)
    else:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError as error:
                raise SystemExit(f"Could not stop PBrainZ process {pid}: {error}") from error

    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        remaining = set(_running_pbrainz_pids()).intersection(pids)
        if not remaining:
            print("Existing PBrainZ process(es) stopped.")
            return pids
        time.sleep(0.1)

    remaining = tuple(sorted(set(_running_pbrainz_pids()).intersection(pids)))
    if remaining:
        print(f"Forcing remaining PBrainZ process(es): {', '.join(map(str, remaining))}")
        if os.name == "nt":
            for pid in remaining:
                _taskkill(pid, force=True)
        else:
            for pid in remaining:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except PermissionError as error:
                    raise SystemExit(
                        f"Could not force-stop PBrainZ process {pid}: {error}"
                    ) from error
        final_remaining = tuple(sorted(set(_running_pbrainz_pids()).intersection(pids)))
        if final_remaining:
            raise SystemExit(
                "PBrainZ process(es) did not exit: "
                + ", ".join(map(str, final_remaining))
            )
    print("Existing PBrainZ process(es) stopped.")
    return pids


def _running_pbrainz_pids() -> list[int]:
    """Return PBrainZ PIDs without broad name or command-line matching."""
    if os.name == "nt":
        return _windows_pbrainz_pids()
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Could not inspect running processes: {error}") from error
    pids: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        comm = fields[1]
        args = fields[2] if len(fields) == 3 else ""
        if _is_pbrainz_process(comm, args):
            pids.append(pid)
    return pids


def _windows_pbrainz_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Could not inspect running processes: {error}") from error
    pids: list[int] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2 or not _is_pbrainz_process_name(row[0]):
            continue
        try:
            pids.append(int(row[1]))
        except ValueError:
            continue
    return pids


def _is_pbrainz_process(comm: str, args: str) -> bool:
    """Match a PBrainZ binary/AppImage, without matching project paths."""
    if _is_pbrainz_process_name(comm):
        return True
    try:
        tokens = shlex.split(args, posix=os.name != "nt")
    except ValueError:
        tokens = args.split()
    if not tokens:
        return False
    if _is_pbrainz_process_name(Path(tokens[0]).name):
        return True
    return "-m" in tokens and any(
        token.casefold() == PRODUCT_BINARY_NAME.casefold() for token in tokens[1:]
    )


def _is_pbrainz_process_name(value: str) -> bool:
    name = Path(str(value).strip().strip('"')).name.casefold()
    product = PRODUCT_BINARY_NAME.casefold()
    return name in {product, f"{product}.exe"} or (
        name.startswith(f"{product}-") and name.endswith((".appimage", ".exe"))
    )


def _taskkill(pid: int, *, force: bool) -> None:
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    subprocess.run(command, check=False, capture_output=True, text=True)


def _restart_artifact(artifact: Path) -> None:
    """Start a successful artifact detached from the release builder."""
    if not artifact.is_file():
        raise SystemExit(f"Cannot restart missing release artifact: {artifact}")
    popen_kwargs: dict[str, object] = {
        "cwd": artifact.parent,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen([str(artifact)], **popen_kwargs)
    print(f"Restarted PBrainZ from {artifact} (pid={process.pid}).")


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
