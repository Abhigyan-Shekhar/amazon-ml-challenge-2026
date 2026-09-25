"""Fast edit features, without external data or pretrained parameters."""
try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
except ImportError as exc:
    raise ImportError('Fuzzy model features require RapidFuzz; install requirements.txt in the active environment.') from exc
NAMES=['name_levenshtein','address_levenshtein','name_token_sort','address_token_sort','name_weighted_ratio','address_weighted_ratio']
def fuzzy_features(name,address,target_name,target_address):
    return [Levenshtein.normalized_similarity(name,target_name),Levenshtein.normalized_similarity(address,target_address) if address and target_address else 0,fuzz.token_sort_ratio(name,target_name)/100,fuzz.token_sort_ratio(address,target_address)/100 if address and target_address else 0,fuzz.WRatio(name,target_name)/100,fuzz.WRatio(address,target_address)/100 if address and target_address else 0]
