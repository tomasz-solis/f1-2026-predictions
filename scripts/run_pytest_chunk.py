"""Run pytest for a deterministic alphabetic chunk of test files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _select_test_files(letters: str) -> list[str]:
    selected_letters = {letter.lower() for letter in letters if letter.isalpha()}
    if not selected_letters:
        raise ValueError("--letters must include at least one alphabetic character")

    test_files = sorted(Path("tests").glob("test_*.py"))
    return [
        str(path)
        for path in test_files
        if len(path.name) > len("test_") and path.name[len("test_")].lower() in selected_letters
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--letters", required=True, help="First letters after test_ to include.")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    selected_files = _select_test_files(args.letters)
    if not selected_files:
        print(f"No test files matched chunk letters: {args.letters}", file=sys.stderr)
        return 2

    return subprocess.call([sys.executable, "-m", "pytest", *selected_files, *pytest_args])


if __name__ == "__main__":
    raise SystemExit(main())
