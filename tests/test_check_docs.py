import subprocess
import sys
from pathlib import Path

import pytest

# The checker is a repository script, not part of the installed package, so
# tests run it as the prek hook does: as a subprocess on a synthetic tree.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_docs.py"


def write_files(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run_check(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "files",
    [
        {"README.md": "[Guide](docs/guide.md)\n", "docs/guide.md": "# Guide\n"},
        {
            "README.md": "[Setup](docs/guide.md#one-time-setup)\n",
            "docs/guide.md": "# Guide\n\n## One-Time Setup\n",
        },
        {"README.md": "# Usage\n\n# Usage\n\n[Second](#usage-1)\n"},
        {"README.md": "## `uv` and Git 2.28\n\n[Tools](#uv-and-git-228)\n"},
        {"README.md": "[Folder](docs/)\n", "docs/guide.md": "# Guide\n"},
        {"README.md": "[Site](https://example.invalid/missing.md)\n"},
        {"README.md": "```md\n[Example](missing.md)\n```\n"},
        {"README.md": "Write `[text](missing.md)` for links.\n"},
        {"storage/notes.md": "[Generated](missing.md)\n"},
        {".venv/pkg/README.md": "[Vendored](missing.md)\n"},
    ],
    ids=[
        "relative-file",
        "heading-anchor",
        "duplicate-heading-suffix",
        "punctuation-in-heading",
        "directory",
        "external-url",
        "link-in-code-fence",
        "link-in-inline-code",
        "ignored-data-dir",
        "hidden-dir",
    ],
)
def test_check_docs_accepts_valid_or_skipped_links(
    tmp_path: Path, files: dict[str, str]
) -> None:
    write_files(tmp_path, files)

    result = run_check(tmp_path)

    assert (result.returncode, result.stdout) == (0, "")


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {"docs/guide.md": "See [Missing](../missing.md).\n"},
            "docs/guide.md:1: link target not found: ../missing.md",
        ),
        (
            {"README.md": "# Usage\n\n[Install](#install)\n"},
            "README.md:3: heading anchor not found: #install",
        ),
        (
            {
                "README.md": "[Setup](docs/guide.md#setup)\n",
                "docs/guide.md": "# Guide\n\n```sh\n# setup\n```\n",
            },
            "README.md:1: heading anchor not found: docs/guide.md#setup",
        ),
    ],
    ids=["missing-file", "missing-anchor", "comment-in-fence-is-not-heading"],
)
def test_check_docs_reports_broken_links(
    tmp_path: Path, files: dict[str, str], expected: str
) -> None:
    write_files(tmp_path, files)

    result = run_check(tmp_path)

    assert (result.returncode, result.stdout.strip()) == (1, expected)


def test_check_docs_reports_readme_over_line_budget(tmp_path: Path) -> None:
    write_files(tmp_path, {"README.md": "line\n" * 4})

    result = run_check(tmp_path, "--readme-max-lines", "3")

    assert result.returncode == 1
    assert result.stdout.startswith("README.md: 4 lines exceeds the 3-line budget")


def test_check_docs_accepts_readme_at_line_budget(tmp_path: Path) -> None:
    write_files(tmp_path, {"README.md": "line\n" * 3})

    result = run_check(tmp_path, "--readme-max-lines", "3")

    assert result.returncode == 0
