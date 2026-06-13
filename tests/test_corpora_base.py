from eval.corpora.base import Segment, Reference


def test_per_speaker_text_groups_and_orders_by_time():
    ref = Reference(uri="m1", segments=[
        Segment(speaker="A", start=0.0, end=1.0, text="hello world"),
        Segment(speaker="B", start=1.0, end=2.0, text="hi"),
        Segment(speaker="A", start=2.0, end=3.0, text="again"),
    ])
    assert ref.per_speaker_text() == {"A": "hello world again", "B": "hi"}


def test_to_rttm_lines_format():
    ref = Reference(uri="m1", segments=[Segment("A", 0.0, 1.5, "x")])
    line = ref.to_rttm().strip()
    # SPEAKER <uri> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
    assert line == "SPEAKER m1 1 0.000 1.500 <NA> <NA> A <NA> <NA>"
