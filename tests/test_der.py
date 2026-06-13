import textwrap
from eval.metrics.der import diarization_error_rate


def test_identical_rttm_zero(tmp_path):
    rttm = textwrap.dedent(
        """\
        SPEAKER m1 1 0.000 5.000 <NA> <NA> A <NA> <NA>
        SPEAKER m1 1 5.000 5.000 <NA> <NA> B <NA> <NA>
        """
    )
    ref = tmp_path / "ref.rttm"; ref.write_text(rttm)
    hyp = tmp_path / "hyp.rttm"; hyp.write_text(rttm)
    assert diarization_error_rate(str(ref), str(hyp)) == 0.0
