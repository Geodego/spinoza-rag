"""Check repository Markdown for broken relative links and README growth.

Scans every Markdown file under the root, skipping hidden directories and the
ignored local data directories. Reports each relative link whose target file
or heading anchor does not exist, and a root README.md longer than the line
budget. External URLs are not fetched. Prints one ``path:line: message`` per
problem and exits with status 1 when any is found.
"""

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path

README_MAX_LINES = 250
SKIPPED_DIRS = frozenset({"data", "dist", "models", "storage"})

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_INLINE_CODE = re.compile(r"`[^`]*`")
_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def markdown_files(root: Path) -> list[Path]:
    """Return the Markdown files under root, excluding skipped directories."""
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(
            part.startswith(".") or part in SKIPPED_DIRS
            for part in path.relative_to(root).parts[:-1]
        )
    )


def prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield one-based line numbers and lines outside fenced code blocks."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            yield number, line


def heading_slug(heading: str) -> str:
    """Return GitHub's anchor slug for a heading, before duplicate suffixes."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"[`*]", "", text).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


def anchors(text: str) -> set[str]:
    """Return the heading anchors GitHub generates for a Markdown document.

    Repeated headings receive ``-1``, ``-2``, ... suffixes, as on GitHub.
    """
    seen: dict[str, int] = {}
    result = set()
    for _, line in prose_lines(text):
        match = _HEADING.match(line)
        if match:
            slug = heading_slug(match.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def link_problems(path: Path, root: Path) -> Iterator[str]:
    """Yield a message for each broken relative link or anchor in a file."""
    text = path.read_text(encoding="utf-8")
    for number, line in prose_lines(text):
        for target in _LINK.findall(_INLINE_CODE.sub("", line)):
            if _SCHEME.match(target):
                continue
            file_part, _, anchor = target.partition("#")
            destination = (path.parent / file_part).resolve() if file_part else path
            location = f"{path.relative_to(root)}:{number}"
            if not destination.exists():
                yield f"{location}: link target not found: {target}"
            elif anchor and destination.suffix == ".md" and destination.is_file():
                if anchor not in anchors(destination.read_text(encoding="utf-8")):
                    yield f"{location}: heading anchor not found: {target}"


def readme_problems(root: Path, max_lines: int) -> Iterator[str]:
    """Yield a message when the root README exceeds its line budget."""
    readme = root / "README.md"
    if readme.is_file():
        count = len(readme.read_text(encoding="utf-8").splitlines())
        if count > max_lines:
            yield (
                f"README.md: {count} lines exceeds the {max_lines}-line budget; "
                "move detail to a docs/ page and link to it"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the checks and return the process exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="repository root to scan"
    )
    parser.add_argument(
        "--readme-max-lines",
        type=int,
        default=README_MAX_LINES,
        help="maximum number of lines in the root README.md",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    problems = list(readme_problems(root, args.readme_max_lines))
    for path in markdown_files(root):
        problems.extend(link_problems(path, root))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
