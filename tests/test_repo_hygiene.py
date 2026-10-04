"""Guard rails: credentials must never be tracked by git (and so never pushed)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CREDENTIAL_NAME = re.compile(r"(^|/)(\.env[^/]*|[^/]*\.env|[^/]*credentials[^/]*|[^/]*\.pem|\.pg_service\.conf)$",
                             re.IGNORECASE)
# a connection URL with an inline password, other than an obvious placeholder
PASSWORD_URL = re.compile(r"postgres(?:ql)?://[^:/@\s\"'`]+:(?!\*\*\*\*|<|\{|\$|password@|pass@|b@)[^@\s\"'`]+@")

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                                reason="needs a git checkout")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def test_no_credential_files_are_tracked():
    tracked = _git("ls-files").splitlines()
    assert [f for f in tracked if CREDENTIAL_NAME.search(f)] == []


def test_no_tracked_file_contains_a_password_url():
    hits = []
    for f in _git("ls-files").splitlines():
        path = ROOT / f
        if path.suffix in {".json", ".png", ".pdf", ".pt", ".safetensors"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f for _ in PASSWORD_URL.finditer(text)]
    assert hits == []


def test_tiger_credential_downloads_are_ignored():
    for name in ("tiger-cloud-db-inference-credentials.env", "tiger-cloud-db-inference-credentials.txt",
                 ".env", "local.env"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=ROOT).returncode == 0, name
