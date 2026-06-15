import threading
from tathurell.naming_ui import create_app


def _samples():
    return {
        "SPEAKER_00": {"start": 0.0, "end": 8.0, "text": "hello there"},
        "SPEAKER_01": {"start": 9.0, "end": 12.0, "text": "good morning"},
    }


def test_index_renders_field_and_audio_per_speaker(tmp_path):
    app = create_app(_samples(), str(tmp_path), {}, threading.Event())
    html = app.test_client().get("/").get_data(as_text=True)
    for spk in ("SPEAKER_00", "SPEAKER_01"):
        assert f'name="{spk}"' in html
        assert f"/clip/{spk}" in html
    assert "hello there" in html


def test_submit_fills_result_and_sets_done(tmp_path):
    result, done = {}, threading.Event()
    app = create_app(_samples(), str(tmp_path), result, done)
    app.test_client().post("/submit", data={"SPEAKER_00": "Dave", "SPEAKER_01": "  "})
    assert result == {"SPEAKER_00": "Dave", "SPEAKER_01": "SPEAKER_01"}
    assert done.is_set()


def test_clip_unknown_speaker_404(tmp_path):
    app = create_app(_samples(), str(tmp_path), {}, threading.Event())
    assert app.test_client().get("/clip/../etc/passwd").status_code == 404
    assert app.test_client().get("/clip/SPEAKER_99").status_code == 404


def test_transcript_text_is_html_escaped(tmp_path):
    # Whisper can emit angle brackets (e.g. <unk>); they must be escaped, not
    # rendered as tags, or the modal breaks.
    samples = {"SPEAKER_00": {"start": 0.0, "end": 1.0, "text": "say <unk> & go"}}
    app = create_app(samples, str(tmp_path), {}, threading.Event())
    html = app.test_client().get("/").get_data(as_text=True)
    assert "&lt;unk&gt;" in html and "&amp;" in html
    assert "<unk>" not in html
