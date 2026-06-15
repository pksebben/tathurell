"""Sentence-level speaker realignment (fixes intra-sentence speaker flips).

WhisperX assigns each word a speaker independently by max-overlap, so a single
word at a turn boundary can flip to a wrong/short neighbour turn and fragment one
person's sentence (e.g. "out [SPK1] of their [SPK2] minds [SPK1]"). The accepted
community fix is to reassign speakers per *sentence* — a punctuation-delimited
span — by majority vote, guarded so a span with no clear majority is left alone
(genuine fast exchanges aren't force-merged).

The three functions below and the punctuation constant are vendored VERBATIM from
MahmoudAshraf97/whisper-diarization (helpers.py). They are pure stdlib and operate
on a list of dicts each carrying at least "word" (a punctuated token) and
"speaker"; any other keys (e.g. start/end) are copied through unchanged.

----------------------------------------------------------------------------
Vendored from https://github.com/MahmoudAshraf97/whisper-diarization (helpers.py)

BSD 2-Clause License — Copyright (c) 2023, Mahmoud Ashraf

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
----------------------------------------------------------------------------
"""

sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return right_idx if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx) else -1


def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (right_idx - left_idx + 1)
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1

    return realigned_list


# --- end vendored code ---


def realign_speakers(words):
    """Return `words` with per-sentence-majority-corrected speakers.

    `words` is our standard word list — dicts with at least "word" (punctuated
    token) and "speaker"; "start"/"end" are preserved. Words with an empty
    "word" are passed through untouched (the vendored code indexes word[-1]).
    Non-mutating: operates on copies.
    """
    indexed = [(i, w) for i, w in enumerate(words) if w.get("word")]
    realigned = get_realigned_ws_mapping_with_punctuation([w for _, w in indexed])
    out = [dict(w) for w in words]
    for (orig_i, _), fixed in zip(indexed, realigned):
        out[orig_i] = fixed
    return out
