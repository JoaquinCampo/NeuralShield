from neuralshield.preprocessing.pipeline import preprocess


def test_crlf_does_not_trigger_badcrlf() -> None:
    raw = "GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
    out = preprocess(raw)
    assert "BADCRLF" not in out


def test_query_double_encoding_preserves_percent_space_and_marks_doublepct() -> None:
    raw = "GET /?enc=a%2520b HTTP/1.1\nHost: example.com\n\n"
    out = preprocess(raw)
    assert "enc=a%20b" in out
    assert "DOUBLEPCT" in out
    assert "PCTSPACE" in out
    assert "enc=a b" not in out


def test_query_percent_nul_is_not_emitted_as_literal_nul() -> None:
    raw = "GET /?nul=%00 HTTP/1.1\nHost: example.com\n\n"
    out = preprocess(raw)
    assert "nul=%00" in out
    assert "\x00" not in out


def test_header_flags_are_space_separated_not_csv() -> None:
    raw = (
        "GET / HTTP/1.1\n"
        "Host: example.com\n"
        "Accept: text/html\n"
        "Accept: application/xml\n"
        "\n"
    )
    out = preprocess(raw)
    assert "DUPHDR,HDRMERGE" not in out
    assert "DUPHDR" in out
    assert "HDRMERGE" in out


def test_qsep_flags_are_in_aggregate_flags() -> None:
    raw = "GET /?a=1;b=2&c=3 HTTP/1.1\nHost: example.com\n\n"
    out = preprocess(raw)
    # Both should appear in [QSEP] and also be aggregated into [FLAGS].
    assert "QSEMISEP" in out


def test_mixedsep_from_entity_is_aggregated() -> None:
    raw = "GET /?q=a&#x26;b&x=1 HTTP/1.1\nHost: example.com\n\n"
    out = preprocess(raw)
    assert "[QSEP]" in out
    assert "MIXEDSEP" in out
    assert "QRAWSEMI" in out


def test_orphan_obs_fold_sets_badhdrcont_global_flag() -> None:
    raw = "GET / HTTP/1.1\n orphan-continuation\nHost: example.com\n\n"
    out = preprocess(raw)
    assert "BADHDRCONT" in out
