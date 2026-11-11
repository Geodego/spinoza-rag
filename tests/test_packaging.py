import subprocess
import sys
from pathlib import Path


def test_package_imports_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import spinoza_rag; "
            "from importlib.metadata import metadata; "
            "assert metadata('spinoza-rag')['Name'] == 'spinoza-rag'",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
