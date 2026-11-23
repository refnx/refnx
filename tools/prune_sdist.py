#!/usr/bin/env python3
"""
Remove files/directories that shouldn't be shipped in the sdist.

Run by ``meson dist`` (and hence by meson-python when building an sdist) via
``meson.add_dist_script``. Meson stages a copy of the source tree in
``MESON_DIST_ROOT`` before archiving it; this script prunes that copy.
"""

import os
import shutil
from pathlib import Path

EXCLUDE = [
    # directories
    "web",
    "tools/app",
    "tools/slim",
    "benchmarks",
    "doc",
    ".github",
    "paper",
    "experimental",
    "examples",
    # files
    "readthedocs.yml",
    ".pre-commit-config.yaml",
    ".requirements.txt",
    "testimonials.bib",
    "tox.ini",
    "release_notes.txt",
    ".gitignore",
]


def main():
    root = Path(os.environ["MESON_DIST_ROOT"])
    for rel in EXCLUDE:
        p = root / rel
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()


if __name__ == "__main__":
    main()
