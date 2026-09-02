#!/usr/bin/env python3
"""Render the canonical PBrainZ SVG into Tk- and desktop-safe PNG assets."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ElementTree.register_namespace("", SVG_NAMESPACE)


def main() -> int:
    arguments = _parse_args()
    source = arguments.source.resolve()
    destination = arguments.output.resolve()
    svg = source.read_text(encoding="utf-8")
    if arguments.transparent:
        svg = _remove_container(svg)

    try:
        import cairosvg
    except ImportError as error:  # pragma: no cover - exercised by build environments.
        raise SystemExit(
            "CairoSVG is required to render PBrainZ icons. "
            "Install the build dependencies with: pip install -e '.[build]'"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(destination),
        output_width=arguments.size,
        output_height=arguments.size,
    )
    print(f"Rendered {source} -> {destination} ({arguments.size}x{arguments.size})")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Canonical SVG source.")
    parser.add_argument("--output", type=Path, required=True, help="PNG destination.")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Square output size in pixels (default: 256).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Remove the rounded background container and keep only the artwork.",
    )
    return parser.parse_args()


def _remove_container(svg: str) -> str:
    root = ElementTree.fromstring(svg)
    for child in list(root):
        if _local_name(child.tag) != "rect":
            continue
        if child.get("width") == "128" and child.get("height") == "128":
            root.remove(child)
            return ElementTree.tostring(root, encoding="unicode")
    raise SystemExit(
        "The canonical PBrainZ SVG does not contain its expected background container."
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


if __name__ == "__main__":
    raise SystemExit(main())
