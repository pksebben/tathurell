import importlib


def test_no_ui_uses_prompt_names(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")
    monkeypatch.setattr(cli, "prompt_names", lambda groups: {"_": "from_prompt"})
    out = cli.resolve_names(words=[], groups=[{"speaker": "A", "text": "x"}],
                            audio_path="a.wav", no_ui=True)
    assert out == {"_": "from_prompt"}


def test_ui_failure_falls_back_to_prompt(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")
    monkeypatch.setattr(cli, "prompt_names", lambda groups: {"_": "fallback"})

    def boom(*a, **k):
        raise RuntimeError("no display")

    import tathurell.naming_ui as ui
    monkeypatch.setattr(ui, "collect_names", boom)
    out = cli.resolve_names(words=[{"word": "x", "start": 0.0, "end": 1.0, "speaker": "A"}],
                            groups=[{"speaker": "A", "text": "x"}],
                            audio_path="a.wav", no_ui=False)
    assert out == {"_": "fallback"}
