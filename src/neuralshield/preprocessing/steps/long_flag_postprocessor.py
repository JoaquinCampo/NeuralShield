"""Experimental post-processor that stretches specific flag tokens."""

from __future__ import annotations

from neuralshield.preprocessing.http_preprocessor import HttpPreprocessor


class LongFlagTokenAugmenter(HttpPreprocessor):
    """
    Replace selected flag tokens with exaggerated variants.

    This is intended for isolated experiments that study how downstream
    tokenisers react to unusually long flag identifiers.
    """

    FLAG_MAPPING: dict[str, str] = {
        "FLAG_RISK_HIGH": "FLAG_RISK_HIGH_SUPERFLAGTOKEN_EXPERIMENTAL",
        "QUOTE": "QUOTE_SUPERFLAGTOKEN_EXPERIMENTAL",
        "SEMICOLON": "SEMICOLON_SUPERFLAGTOKEN_EXPERIMENTAL",
        "QSQLI_QUOTE_SEMI": "QSQLI_QUOTE_SEMI_SUPERFLAGTOKEN_EXPERIMENTAL",
        "QRAWSEMI": "QRAWSEMI_SUPERFLAGTOKEN_EXPERIMENTAL",
        "ANGLE": "ANGLE_SUPERFLAGTOKEN_EXPERIMENTAL",
        "XSS_TAG": "XSS_TAG_SUPERFLAGTOKEN_EXPERIMENTAL",
        "PIPE": "PIPE_SUPERFLAGTOKEN_EXPERIMENTAL",
        "PCTSPACE": "PCTSPACE_SUPERFLAGTOKEN_EXPERIMENTAL",
        "QNUL": "QNUL_SUPERFLAGTOKEN_EXPERIMENTAL",
    }

    def process(self, request: str) -> str:
        transformed_lines: list[str] = []

        for line in request.split("\n"):
            if not line:
                transformed_lines.append(line)
                continue

            tokens = line.split()
            if not tokens:
                transformed_lines.append(line)
                continue

            start_idx = 1 if tokens[0].startswith("[") else 0
            prefix = tokens[:start_idx]
            flag_tokens = tokens[start_idx:]

            updated_tokens = [self._transform_token(token) for token in flag_tokens]

            transformed_lines.append(" ".join(prefix + updated_tokens))

        return "\n".join(transformed_lines)

    def _transform_token(self, token: str) -> str:
        base, sep, suffix = token.partition(":")

        mapped = self.FLAG_MAPPING.get(base, base)
        if sep:
            return f"{mapped}{sep}{suffix}"
        return mapped
