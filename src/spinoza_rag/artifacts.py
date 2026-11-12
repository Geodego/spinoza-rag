"""Write deterministic local artifacts without replacing unrelated results."""

import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# A decoded JSON object whose fields the loading stage validates at run time.
type JsonObject = dict[str, Any]


def json_bytes(value: object) -> bytes:
    """Encode a readable, deterministic JSON artifact."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    """Encode one deterministic UTF-8 JSON object per line."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    ).encode("utf-8")


def publish_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    """Publish a complete directory or reuse byte-identical existing artifacts.

    Files are written to a sibling temporary directory that is renamed into
    place, so readers never see a partial result. Existing content is never
    modified.

    Raises:
        FileExistsError: The destination exists with different or incomplete
            contents.
    """
    if output_dir.is_symlink():
        raise ValueError("The output directory must not be a symlink.")
    if output_dir.exists():
        if (
            output_dir.is_dir()
            and {p.name for p in output_dir.iterdir()} == set(artifacts)
            and all(
                not (output_dir / name).is_symlink()
                and (output_dir / name).is_file()
                and (output_dir / name).read_bytes() == data
                for name, data in artifacts.items()
            )
        ):
            return
        raise FileExistsError(
            f"{output_dir} already exists with different or incomplete contents; "
            "choose a new --output-dir. Existing files were preserved."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        for name, data in artifacts.items():
            (staging / name).write_bytes(data)
        staging.rename(output_dir)


def save_json(path: Path, value: object) -> None:
    """Atomically replace a pipeline-owned checkpoint file."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
