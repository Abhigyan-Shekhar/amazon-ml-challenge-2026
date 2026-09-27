"""Train-fitted, versioned matching features shared by training and inference.

Alternate views never modify retrieval or the production normalization policy.
Call fit with training S1 rows only, then persist the extractor with the model.
"""
from collections import Counter
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import re
from pathlib import Path
import unicodedata

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from src.normalize import normalize, VERSION as NORMALIZATION_VERSION
from src.features.fuzzy import fuzzy_features, NAMES as FUZZY_NAMES

VERSION = 'matching-features-v1'
BASE_NAMES = ['name_jaccard', 'address_jaccard', 'name_exact', 'address_exact',
              'numeric_jaccard', 'numeric_conflict', 'country_match', 'name_length_ratio',
              'address_length_ratio', 'address_missing', *FUZZY_NAMES]
FAMILY_NAMES = {
    'name': ['name_idf_jaccard', 'name_idf_containment_min', 'name_idf_containment_max',
             'name_distinctive_conflict', 'name_core_exact', 'name_core_token_sort',
             'name_core_jaccard', 'name_acronym_match', 'name_initials_match',
             'name_idf_soft_agreement'],
    'address': ['postcode_equal', 'postcode_conflict', 'postcode_both_known',
                'building_equal', 'building_conflict', 'building_both_known',
                'unit_equal', 'unit_conflict', 'unit_both_known', 'address_word_jaccard',
                'address_word_token_sort', 'address_word_idf_jaccard',
                'address_abbreviation_token_sort'],
    'views': ['name_romanized_token_sort', 'name_romanized_levenshtein',
              'address_romanized_token_sort', 'name_romanization_changed',
              'name_ocr_token_sort', 'name_ocr_changed', 'name_compact_similarity',
              'name_nonlatin_mismatch'],
}
LEGAL_SUFFIXES = frozenset({'limited', 'private', 'pvt', 'ltd', 'llc', 'inc',
                           'incorporated', 'corp', 'corporation', 'llp', 'co', 'company'})
ADDRESS_ALIASES = {'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
                   'hwy': 'highway', 'ln': 'lane', 'dr': 'drive', 'apt': 'apartment',
                   'ste': 'suite'}
NAN = float('nan')


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def baseline_features(qn, qa, qc, tn, ta, tc):
    """Legacy v2 16-column contract, taking already normalized strings."""
    qnt, tnt, qat, tat = set(qn.split()), set(tn.split()), set(qa.split()), set(ta.split())
    qnum = {t for t in qat if any(c.isdigit() for c in t)}
    tnum = {t for t in tat if any(c.isdigit() for c in t)}
    ratio = lambda a, b: min(len(a), len(b)) / max(1, len(a), len(b))
    values = [jaccard(qnt, tnt), jaccard(qat, tat), float(qn == tn),
              float(bool(qa) and qa == ta), jaccard(qnum, tnum),
              float(bool(qnum and tnum) and not qnum & tnum), float(qc == tc),
              ratio(qn, tn), ratio(qa, ta), float(not qa or not ta)]
    if not qa or not ta:
        for i in (1, 3, 4, 5, 8):
            values[i] = NAN
    return values + fuzzy_features(qn, qa, tn, ta, neutral_missing_address=True)


def core_words(name):
    words = name.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return words


def country_code(country):
    return {'us': 'us', 'usa': 'us', 'united states': 'us', 'india': 'in',
            'in': 'in', 'france': 'fr', 'fr': 'fr'}.get(country, 'unknown')


def number_view(value):
    return re.sub(r'\d+', lambda m: str(int(m.group())), value.lower()) if value else None


@lru_cache(maxsize=32768)
def address_parts(raw, country):
    """Conservative heuristics, not a general parser; ambiguous components are unknown."""
    raw = unicodedata.normalize('NFKC', raw).lower().strip()
    code = country_code(country)
    if code == 'us':
        post = re.findall(r'\b[a-z]{2}\s+(\d{5})(?:-\d{4})?\s*$', raw)
    elif code == 'in':
        post = re.findall(r'\b([1-9]\d{5})\b', raw)
    elif code == 'fr':
        post = re.findall(r'\b(\d{5})\s+[a-zà-ÿ]', raw)
    else:
        post = []
    postcode = next(iter(set(post))) if len(set(post)) == 1 else None
    building = re.match(r'^(\d{1,5}[a-z]?)(?=\s|,|$)', raw) if code != 'unknown' else None
    # Avoid interpreting the first half of an explicitly separated range as a number.
    if re.match(r'^\d+[a-z]?\s*[-/]\s*\d+', raw):
        building = None
    units = re.findall(r'\b(?:unit|suite|ste|apartment|apt|flat|shop)\.?\s*#?\s*([a-z0-9]+)\b', raw)
    unit = next(iter(set(units))) if len(set(units)) == 1 else None
    words = normalize(raw).split()
    expanded = [ADDRESS_ALIASES.get(w, w) for w in words] if code in {'us', 'in'} else words
    alpha = tuple(w for w in expanded if w.isalpha())
    return postcode, number_view(building.group(1)) if building else None, number_view(unit), alpha, ' '.join(expanded)


def comparison(a, b):
    return [float(a == b), float(a != b), 1.] if a is not None and b is not None else [NAN, NAN, 0.]


def ocr_view(text):
    # Alternate name view only. Numeric address fields are never passed here.
    return ' '.join(w.translate(str.maketrans('015', 'ols')) if any(c.isalpha() for c in w)
                    else w for w in text.split())


def nonlatin(text):
    return any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text)


@lru_cache(maxsize=32768)
def romanize(text):
    from anyascii import anyascii
    return normalize(anyascii(unicodedata.normalize('NFKC', text)))


class PairFeatureExtractor:
    def __init__(self, families=()):
        if len(set(families)) != len(families) or set(families) - set(FAMILY_NAMES):
            raise ValueError('Unknown or duplicate feature family')
        self.families = tuple(f for f in FAMILY_NAMES if f in families)
        if 'views' in self.families and version('anyascii') != '0.3.3':
            raise ValueError('views requires anyascii==0.3.3 for reproducible mappings')
        self.name_df, self.address_df = Counter(), Counter()
        self.document_count = 0
        self.train_ids_sha256 = None

    @property
    def feature_names(self):
        return BASE_NAMES + [n for f in self.families for n in FAMILY_NAMES[f]]

    def fit(self, train_records):
        """Fit DF on unique training S1s only; never reads labels or target rows."""
        names, addresses, ids = Counter(), Counter(), set()
        for row in train_records:
            eid = row['entity_id']
            if eid in ids:
                raise ValueError(f'Duplicate training S1: {eid}')
            ids.add(eid)
            names.update(set(normalize(row['business_name']).split()))
            addresses.update(set(normalize(row['business_address']).split()))
        if not ids:
            raise ValueError('Training records must not be empty')
        self.name_df, self.address_df, self.document_count = names, addresses, len(ids)
        self.train_ids_sha256 = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
        return self

    def _weight(self, token, address=False):
        df = self.address_df if address else self.name_df
        return math.log((1 + self.document_count) / (1 + df.get(token, 0))) + 1

    def _overlap(self, a, b, address=False):
        if not a or not b:
            return [NAN] * 4
        weights = {t: self._weight(t, address) for t in a | b}
        shared = sum(weights[t] for t in sorted(a & b))
        ca = shared / sum(weights[t] for t in sorted(a))
        cb = shared / sum(weights[t] for t in sorted(b))
        conflict = max((weights[t] for t in a ^ b), default=0) / max(weights.values())
        return [shared / sum(weights[t] for t in sorted(weights)), min(ca, cb), max(ca, cb), conflict]

    def _name_features(self, qn, tn):
        a, b = core_words(qn), core_words(tn)
        aset, bset = set(a), set(b)
        values = self._overlap(set(qn.split()), set(tn.split()))
        if not a or not b:
            return values + [NAN] * 6
        ai, bi = ''.join(w[0] for w in a), ''.join(w[0] for w in b)
        acronym = (len(a) > 1 and ai in bset) or (len(b) > 1 and bi in aset)
        aa, bb = sorted(aset)[:32], sorted(bset)[:32]
        def soft(left, right):
            return sum(self._weight(w) * max(fuzz.ratio(w, v) / 100 for v in right)
                       for w in left) / sum(self._weight(w) for w in left)
        return values + [float(a == b), fuzz.token_sort_ratio(' '.join(a), ' '.join(b)) / 100,
                         jaccard(aset, bset), float(acronym), float(ai == bi),
                         (soft(aa, bb) + soft(bb, aa)) / 2]

    def _address_features(self, raw_q, raw_t, qc, tc):
        a, b = address_parts(raw_q, qc), address_parts(raw_t, tc)
        postcode = comparison(a[0], b[0]) if country_code(qc) == country_code(tc) else comparison(None, None)
        values = postcode + comparison(a[1], b[1]) + comparison(a[2], b[2])
        if not raw_q.strip() or not raw_t.strip():
            return values + [NAN] * 4
        aw, bw = set(a[3]), set(b[3])
        return values + [jaccard(aw, bw) if aw and bw else NAN,
                         fuzz.token_sort_ratio(' '.join(a[3]), ' '.join(b[3])) / 100 if aw and bw else NAN,
                         self._overlap(aw, bw, address=True)[0],
                         fuzz.token_sort_ratio(a[4], b[4]) / 100]

    def _view_features(self, qn, qa, tn, ta, raw_names_addresses):
        # Transliterate raw NFKC text before punctuation normalization removes
        # combining vowel signs from Indic scripts. Base features remain unchanged.
        qr, tr, qar, tar = map(romanize, raw_names_addresses)
        qo, to = ocr_view(qn), ocr_view(tn)
        return [fuzz.token_sort_ratio(qr, tr) / 100 if qr and tr else NAN,
                Levenshtein.normalized_similarity(qr, tr) if qr and tr else NAN,
                fuzz.token_sort_ratio(qar, tar) / 100 if qar and tar else NAN,
                float(qr != qn or tr != tn), fuzz.token_sort_ratio(qo, to) / 100 if qo and to else NAN,
                float(qo != qn or to != tn),
                Levenshtein.normalized_similarity(qn.replace(' ', ''), tn.replace(' ', '')) if qn and tn else NAN,
                float(nonlatin(qn) != nonlatin(tn))]

    def transform(self, pairs, queries):
        """One float32 batch in input row order. DF never changes during inference."""
        if self.train_ids_sha256 is None:
            raise ValueError('Fit or load the extractor before transform')
        result = []
        for pair in pairs:
            query = queries[pair['source1_entity_id']]
            qn, qa, qc = (normalize(query[k]) for k in ('business_name', 'business_address', 'country'))
            tn, ta, tc = (normalize(pair[k]) for k in ('target_name', 'target_address', 'target_country'))
            values = baseline_features(qn, qa, qc, tn, ta, tc)
            for family in self.families:
                if family == 'name':
                    values += self._name_features(qn, tn)
                elif family == 'address':
                    values += self._address_features(query['business_address'], pair['target_address'], qc, tc)
                else:
                    values += self._view_features(qn, qa, tn, ta,
                        (query['business_name'], pair['target_name'],
                         query['business_address'], pair['target_address']))
            result.append(values)
        return np.asarray(result, dtype=np.float32).reshape(-1, len(self.feature_names))

    def save(self, path):
        if self.train_ids_sha256 is None:
            raise ValueError('Cannot save an unfitted extractor')
        payload = {'version': VERSION, 'normalization_version': NORMALIZATION_VERSION,
                   'families': self.families, 'feature_names': self.feature_names,
                   'document_count': self.document_count, 'train_ids_sha256': self.train_ids_sha256,
                   'name_df': dict(self.name_df), 'address_df': dict(self.address_df),
                   'transliteration': 'anyascii==0.3.3' if 'views' in self.families else None}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload['version'] != VERSION or payload['normalization_version'] != NORMALIZATION_VERSION:
            raise ValueError('Feature/normalization version mismatch')
        obj = cls(payload['families'])
        if payload['feature_names'] != obj.feature_names:
            raise ValueError('Ordered feature schema mismatch')
        if payload.get('transliteration') != ('anyascii==0.3.3' if 'views' in obj.families else None):
            raise ValueError('Transliteration configuration mismatch')
        obj.document_count, obj.train_ids_sha256 = payload['document_count'], payload['train_ids_sha256']
        obj.name_df, obj.address_df = Counter(payload['name_df']), Counter(payload['address_df'])
        return obj
