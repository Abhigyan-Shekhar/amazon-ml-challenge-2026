"""Fast edit features, without external data or pretrained parameters."""
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
except ImportError:  # keep fixture/tests usable before optional dependency install
    from difflib import SequenceMatcher

    class _Levenshtein:
        @staticmethod
        def normalized_similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

    class _Fuzz:
        @staticmethod
        def token_sort_ratio(a, b):
            return 100 * SequenceMatcher(None, ' '.join(sorted(a.split())), ' '.join(sorted(b.split()))).ratio()

        @staticmethod
        def WRatio(a, b):
            return 100 * SequenceMatcher(None, a, b).ratio()

    Levenshtein, fuzz = _Levenshtein, _Fuzz()
NAMES=['name_levenshtein','address_levenshtein','name_token_sort','address_token_sort','name_weighted_ratio','address_weighted_ratio']
def fuzzy_features(name,address,target_name,target_address):
    return [Levenshtein.normalized_similarity(name,target_name),Levenshtein.normalized_similarity(address,target_address) if address and target_address else 0,fuzz.token_sort_ratio(name,target_name)/100,fuzz.token_sort_ratio(address,target_address)/100 if address and target_address else 0,fuzz.WRatio(name,target_name)/100,fuzz.WRatio(address,target_address)/100 if address and target_address else 0]
