"""CSIC-specific preprocessing pipeline with aggressive heuristics enabled."""

from __future__ import annotations

from neuralshield.preprocessing.pipeline import PreprocessorPipeline, resolve

STEP_PATHS = [
    "neuralshield.preprocessing.steps.00_framing_cleanup:FramingCleanup",
    "neuralshield.preprocessing.steps.01_request_structurer:RequestStructurer",
    "neuralshield.preprocessing.steps.02_header_unfold_obs_fold:HeaderUnfoldObsFold",
    "neuralshield.preprocessing.steps.03_header_normalization_duplicates:HeaderNormalizationDuplicates",
    "neuralshield.preprocessing.steps.04_whitespace_collapse:WhitespaceCollapse",
    "neuralshield.preprocessing.steps.05_dangerous_characters_script_mixing:DangerousCharactersScriptMixingCsic",
    "neuralshield.preprocessing.steps.06_absolute_url_builder:AbsoluteUrlBuilder",
    "neuralshield.preprocessing.steps.07_unicode_nkfc_and_control:UnicodeNFKCAndControl",
    "neuralshield.preprocessing.steps.08_percent_decode_once:PercentDecodeOnce",
    "neuralshield.preprocessing.steps.09_html_entity_decode_once:HtmlEntityDecodeOnce",
    "neuralshield.preprocessing.steps.10_query_parser_and_flags:QueryParserAndFlagsCsic",
    "neuralshield.preprocessing.steps.11_path_structure_normalizer:PathStructureNormalizer",
    "neuralshield.preprocessing.steps.12_flag_summary:FlagSummaryEmitterCsic",
]


def build_csic_overfit_pipeline(
    *, max_workers: int | None = None
) -> PreprocessorPipeline:
    """Construct the CSIC-specific overfitted pipeline."""

    steps = [resolve(name) for name in STEP_PATHS]
    return PreprocessorPipeline(steps, max_workers=max_workers)


preprocess_csic_overfit: PreprocessorPipeline = build_csic_overfit_pipeline()
