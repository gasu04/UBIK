"""
Unit tests for ingest/diarization.py — per-file mono/multi detection.

Covers the spec heuristic (fewer than 2 distinct speaker labels => mono),
accent-safe label capture (Ginés/Adrián), the Otter timestamp and markdown
label forms, and prose-colon false-positive suppression.
"""

from ingest.diarization import MONO, MULTI, detect_diarization, speaker_labels


def test_two_distinct_named_speakers_is_multi():
    text = "Ginés: hola qué tal\nAdrián: muy bien gracias\nGinés: me alegro"
    assert detect_diarization(text) == MULTI


def test_single_speaker_is_mono():
    text = "Ginés: hablo\nGinés: y sigo hablando solo\nGinés: todavía yo"
    assert detect_diarization(text) == MONO


def test_no_labels_is_mono():
    text = "Esto es un muro de texto sin ninguna etiqueta de hablante."
    assert detect_diarization(text) == MONO
    assert speaker_labels(text) == set()


def test_accented_names_are_captured():
    # Accent-safe: ASCII [A-Za-z] would miss these; folding makes them distinct.
    text = "Ginés: uno\nSofía: dos\nAdrián: tres"
    labels = speaker_labels(text)
    assert {"gines", "sofia", "adrian"} <= labels
    assert detect_diarization(text) == MULTI


def test_otter_timestamp_format_multi():
    text = "Speaker 1  0:00\nhola a todos\nSpeaker 2  0:05\nbuenas"
    assert detect_diarization(text) == MULTI
    assert {"speaker 1", "speaker 2"} <= speaker_labels(text)


def test_markdown_bold_labels_multi():
    text = "**Leticia:** dime\n**Ginés:** pues mira lo que pasa"
    assert detect_diarization(text) == MULTI


def test_prose_colons_not_counted_as_speakers():
    # "Nota:" / "Fecha:" are line-openers, not speakers -> must stay mono.
    text = "Nota: esto es una nota\nFecha: 2025-06-01\nResumen: breve"
    assert detect_diarization(text) == MONO


def test_accent_folding_collapses_same_name():
    text = "Ginés: uno\ngines: dos\nGINÉS: tres"
    assert speaker_labels(text) == {"gines"}
    assert detect_diarization(text) == MONO


def test_gines_alberto_distinct_from_gines():
    # The son ("Gines Alberto") is a different speaker label from the father.
    text = "Gines: pregunto algo\nGines Alberto: respondo yo"
    assert detect_diarization(text) == MULTI
    assert {"gines", "gines alberto"} <= speaker_labels(text)
