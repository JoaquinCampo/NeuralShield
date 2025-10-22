"""Integration checks for overfitted preprocessing pipelines."""

from __future__ import annotations

from neuralshield.preprocessing.pipeline_csic_overfit import preprocess_csic_overfit
from neuralshield.preprocessing.pipeline_csic_long_flags import (
    preprocess_csic_long_flags,
)
from neuralshield.preprocessing.pipeline_srbh_overfit import preprocess_srbh_overfit


CSIC_ATTACK_WITH_SCRIPT = (
    "GET http://localhost:8080/tienda1/miembros/editar.jsp?"
    "modo=registro%3CSCRIPT%3Ealert%28%22Paros%22%29%3B%3C%2FSCRIPT%3E&"
    "login=widuch&password=o7TOp%E9dIca&nombre=Belindo&apellidos=Calet&"
    "email=gassman%40itrends.gs&dni=54927886E&direccion=Memorial+Nu%F1ez+Blanca+193%2C+&"
    "ciudad=Madridanos&cp=37060&provincia=Teruel&ntc=0916374019101201&B1=Registrar HTTP/1.1\n"
    "User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)\n"
    "Pragma: no-cache\n"
    "Cache-control: no-cache\n"
    "Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5\n"
    "Accept-Encoding: x-gzip, x-deflate, gzip, deflate\n"
    "Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5\n"
    "Accept-Language: en\n"
    "Host: localhost:8080\n"
    "Cookie: JSESSIONID=1428C0B291114BE3069778C24947ADF9\n"
    "Connection: close\n"
)


SRBH_PIPE_ATTACK = (
    "GET /blog/.svn/entries HTTP/1.1\n"
    "Host: test-site.com\n"
    "Cookie: wordpress_logged_in_1aefbe2f76edd740f8e362f39da3353b=rafael%7C1595179514%7C1ldatsgnw2jqMXxJ0LW4HRJoq4C1kZ8VHXV2nlPAKTJ%7C3953326980c143f4ab8dc38e3f9f4d9d8d49fec328cb23e6bd9818490adde67e\n"
    "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0\n"
    "\n"
)


SRBH_PCTSPACE_ATTACK = (
    "HEAD /appServer/jvmReport.jsf?instanceName=server&pageTitle=JVM%20Report HTTP/1.1\n"
    "User-Agent: Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org/book/nse.html)\n"
    "Host: 249.network5-175-62-15.static.network5.net\n"
    "Connection: close\n"
)


def test_csic_overfit_flags() -> None:
    processed = preprocess_csic_overfit(CSIC_ATTACK_WITH_SCRIPT)
    assert "QSQLI_QUOTE_SEMI" in processed
    assert "XSS_TAG" in processed
    assert "FLAG_RISK_HIGH" in processed


def test_srbh_overfit_pipe_repeat() -> None:
    processed = preprocess_srbh_overfit(SRBH_PIPE_ATTACK)
    assert "PIPE_REPEAT" in processed
    assert "STRUCT_GAP:HOME" in processed


def test_srbh_overfit_pctspace_pair_summary() -> None:
    processed = preprocess_srbh_overfit(SRBH_PCTSPACE_ATTACK)
    assert "PCTSPACE_PAIR" in processed
    assert "combo_pctspace=1" in processed


def test_srbh_overfit_multiple_slash_heavy() -> None:
    request = "GET http://example.com///foo//bar HTTP/1.1\nHost: example.com\n\n"
    processed = preprocess_srbh_overfit(request)
    assert "MULTIPLESLASH_HEAVY" in processed
    assert "STRUCT_GAP:HOPBYHOP" in processed


def test_csic_long_flags_expansion() -> None:
    processed = preprocess_csic_long_flags(CSIC_ATTACK_WITH_SCRIPT)
    assert "QSQLI_QUOTE_SEMI_SUPERFLAGTOKEN_EXPERIMENTAL" in processed
    assert "FLAG_RISK_HIGH_SUPERFLAGTOKEN_EXPERIMENTAL" in processed
