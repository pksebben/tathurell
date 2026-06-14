import builtins
import importlib


def test_prompt_names_falls_back_to_label_on_eof(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    groups = [
        {"speaker": "A", "text": "hello"},
        {"speaker": "B", "text": "hi"},
        {"speaker": "A", "text": "again"},  # repeat speaker -> asked once
    ]
    names = cli.prompt_names(groups)
    assert names == {"A": "A", "B": "B"}
