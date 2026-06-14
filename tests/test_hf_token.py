import pytest
from tathurell.whisperx_core import resolve_hf_token


def test_env_token_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert resolve_hf_token(token_file="/nonexistent") == "hf_env"


def test_file_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    f = tmp_path / ".hf_token"
    f.write_text("hf_file\n")
    assert resolve_hf_token(token_file=str(f)) == "hf_file"


def test_missing_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        resolve_hf_token(token_file=str(tmp_path / "absent"))
