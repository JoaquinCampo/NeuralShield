import re
import unicodedata

from loguru import logger

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class RemoveFramingArtifacts(HttpPreprocessor):
    """
    Remove framing artifacts from the HTTP request.
    """

    def _remove_framing_artifacts(self, http_request: str) -> str:
        """
        Remove BOM and control characters from the edges of HTTP request strings.

        This function cleans framing artifacts from the absolute borders
        of the HTTP request string. It removes:
        - BOM (Byte Order Mark) at the beginning
        - Non-printable control characters at the beginning and end

        It preserves all content within the HTTP message structure and
        only modifies the absolute edges to ensure robust parsing downstream.

        Args:
            http_request: Raw HTTP request string that may contain framing artifacts

        Returns:
            HTTP request string with edge artifacts removed
        """

        logger.debug("Removing framing artifacts from HTTP request")

        original_length = len(http_request)
        processed = http_request

        # Remove BOM at the beginning
        bom_removed = False
        if processed.startswith("\ufeff"):
            processed = processed[1:]
            bom_removed = True
            logger.debug("Removed BOM from beginning of HTTP request")

        # Remove control characters from the beginning
        leading_controls = 0
        while (
            processed
            and unicodedata.category(processed[0]) == "Cc"
            and processed[0] not in "\t\r\n"
        ):
            processed = processed[1:]
            leading_controls += 1

        # Remove control characters from the end
        trailing_controls = 0
        while (
            processed
            and unicodedata.category(processed[-1]) == "Cc"
            and processed[-1] not in "\t\r\n"
        ):
            processed = processed[:-1]
            trailing_controls += 1

        # Log what was removed
        if bom_removed or leading_controls > 0 or trailing_controls > 0:
            total_removed = original_length - len(processed)
            logger.info(
                "Pre-parse cleanup: removed {total} chars "
                "(BOM: {bom}, leading controls: {lead}, trailing controls: {trail})",
                total=total_removed,
                bom=bom_removed,
                lead=leading_controls,
                trail=trailing_controls,
            )

        return processed

    def process(self, request: str) -> str:
        return self._remove_framing_artifacts(http_request=request)


class LineStructurePreprocessor(HttpPreprocessor):
    """
    EOL analysis/normalization (headers) and obs-fold unfolding.

    - Detects mixed terminators and bare CR/LF in header section
    - Normalizes header terminators to CRLF
    - Strictly unfolds folded header lines (obs-fold) within headers only
    - Logs anomalies and telemetry via loguru
    """

    def process(self, request: str) -> str:
        if not request:
            return request

        header_lines, _header_end_pos, eol_stats = self._scan_header_lines(request)
        self._log_eol_stats(eol_stats)

        normalized_headers = self._normalize_and_unfold_headers(header_lines)

        # Return only normalized headers, followed by a blank line
        rebuilt = "\r\n".join(normalized_headers) + "\r\n\r\n"
        return rebuilt

    def _scan_header_lines(self, raw: str):
        """
        Scan raw request to extract header lines, header/body split, and EOL stats.

        Returns:
            (header_lines, header_end_pos, eol_stats)
        where header_lines excludes the empty separator line,
        header_end_pos points to the end of string,
        and eol_stats is a dict with counts and flags based on header section.
        """
        crlf_count = 0
        lf_count = 0
        cr_count = 0

        header_lines: list[str] = []
        current: list[str] = []
        i = 0
        n = len(raw)
        header_end_pos = None

        # Helper to finalize current line into header_lines
        def flush_line():
            header_lines.append("".join(current))
            current.clear()

        # Scan until end-of-headers (first empty line). Count EOLs in headers only.
        while i < n:
            ch = raw[i]
            if ch == "\r":
                if i + 1 < n and raw[i + 1] == "\n":
                    crlf_count += 1
                    # End of line
                    if not current:
                        # Empty line -> end of headers at i+2
                        header_end_pos = i + 2
                        i += 2
                        break
                    flush_line()
                    i += 2
                    continue
                else:
                    # Bare CR
                    cr_count += 1
                    if not current:
                        header_end_pos = i + 1
                        i += 1
                        break
                    flush_line()
                    i += 1
                    continue
            elif ch == "\n":
                # Bare LF
                lf_count += 1
                if not current:
                    header_end_pos = i + 1
                    i += 1
                    break
                flush_line()
                i += 1
                continue

            # Normal char within header section
            current.append(ch)
            i += 1

        # If we didn't reach an empty line, we consumed to EOF as headers
        if header_end_pos is None:
            if current:
                flush_line()
            header_end_pos = i

        # EOL flags based on header counts and terminator at EOF/header-end
        kinds_present = sum(1 for c in (crlf_count, lf_count, cr_count) if c > 0)
        eol_mix = kinds_present > 1
        eof_no_crlf = not raw[:header_end_pos].endswith("\r\n\r\n") and not raw[
            :header_end_pos
        ].endswith("\r\n")

        eol_stats = {
            "CRLF": crlf_count,
            "LF": lf_count,
            "CR": cr_count,
            "EOL:MIX": eol_mix,
            "EOL:BARELF": lf_count > 0,
            "EOL:BARECR": cr_count > 0,
            "EOL:EOF_NOCRLF": eof_no_crlf,
        }

        return header_lines, header_end_pos, eol_stats

    def _normalize_and_unfold_headers(self, header_lines: list[str]) -> list[str]:
        """
        Normalize header terminators to CRLF and unfold obs-fold strictly in headers.
        The input is the list of header lines (including request-line).
        """
        if not header_lines:
            return header_lines

        unfolded: list[str] = []
        obs_fold_count = 0
        any_unfold = False

        # RFC7230 tchar for header field-name
        # ! # $ % & ' * + - . ^ _ ` | ~ DIGIT ALPHA

        tchar_re = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

        def is_valid_header(line: str) -> bool:
            if ":" not in line:
                return False
            name, _ = line.split(":", 1)
            return bool(name) and bool(tchar_re.match(name))

        # Preserve request-line as-is at index 0
        unfolded.append(header_lines[0])
        last_was_valid = False

        for idx in range(1, len(header_lines)):
            line = header_lines[idx]
            if line.startswith(" ") or line.startswith("\t"):
                # Continuation line
                if unfolded and last_was_valid:
                    continuation = line.lstrip()
                    unfolded[-1] = (
                        unfolded[-1] + (" " if continuation else "") + continuation
                    )
                    obs_fold_count += 1
                    any_unfold = True
                    continue
                else:
                    # Malformed context for continuation; keep as-is
                    unfolded.append(line)
                    last_was_valid = False
                    continue

            # New header candidate
            if is_valid_header(line):
                unfolded.append(line)
                last_was_valid = True
            else:
                unfolded.append(line)
                last_was_valid = False

        if any_unfold:
            logger.info(
                "OBS-FOLD unfolding applied: {count} continuations",
                count=obs_fold_count,
            )

        return unfolded

    def _log_eol_stats(self, stats: dict) -> None:
        logger.debug(
            "Phase0 EOL stats: CRLF={crlf}, LF={lf}, CR={cr}",
            crlf=stats["CRLF"],
            lf=stats["LF"],
            cr=stats["CR"],
        )
        if stats.get("EOL:MIX") or stats.get("EOL:BARELF") or stats.get("EOL:BARECR"):
            logger.info(
                "Phase0 EOL anomalies: mix={mix}, bareLF={lf}, bareCR={cr}",
                mix=bool(stats.get("EOL:MIX")),
                lf=bool(stats.get("EOL:BARELF")),
                cr=bool(stats.get("EOL:BARECR")),
            )
        if stats.get("EOL:EOF_NOCRLF"):
            logger.info("Phase0 EOL anomaly: header EOF does not end with CRLF")
