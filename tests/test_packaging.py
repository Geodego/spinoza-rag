import subprocess
import sys


def test_package_imports_outside_repository(tmp_path):
    """Run isolated (-I) from another directory, so only an install can import."""
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
