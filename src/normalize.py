import re
import unicodedata

VERSION = "unicode-nfkc-lower-punctuation-space-v1"

# ---------------------------------------------------------------------------
# Diacritic-folding ablation flag — MUST remain False in production.
#
# A capped10-pipeline end-to-end ablation (2026-09-26) showed only a
# +0.00028 macro-F0.5 gain on the disjoint 2,000-S1 confirmation set at the
# frozen 0.585 threshold. The change was therefore REJECTED for production.
# See: artifacts/validation/diacritic_fold_end_to_end.json
#      artifacts/reports/experiment_summary.json key: diacritic_fold_capped10_ablation
#
# DO NOT set this to True in any production or packaging script.
# ---------------------------------------------------------------------------
DIACRITIC_FOLDING_ENABLED: bool = False


def fold_diacritics(value: str) -> str:
    """Remove Latin combining diacritics (e.g. é→e).

    EXPERIMENTAL — for ablation runs only. Never call from the main pipeline.
    Raises RuntimeError if accidentally invoked with the production flag off.
    """
    if not DIACRITIC_FOLDING_ENABLED:
        raise RuntimeError(
            "fold_diacritics() called while DIACRITIC_FOLDING_ENABLED=False. "
            "This function must only be used in isolated ablation scripts."
        )
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"\s+", " ", "".join(c if c.isalnum() or c.isspace() else " " for c in value)).strip()

def prepare(frame):
    frame = frame.copy()
    frame["raw_name"] = frame.business_name
    frame["raw_address"] = frame.business_address
    frame["normalized_name"] = frame.business_name.map(normalize)
    frame["normalized_address"] = frame.business_address.map(normalize)
    frame["normalized_country"] = frame.country.map(normalize)
    frame["retrieval_text"] = frame.normalized_name + " " + frame.normalized_address
    frame["combined_text"] = "[NAME] " + frame.normalized_name + " [ADDRESS] " + frame.normalized_address + " [COUNTRY] " + frame.normalized_country
    return frame
