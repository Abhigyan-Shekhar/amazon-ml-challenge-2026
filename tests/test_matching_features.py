"""Behavioral tests for feature leakage, missingness, schema and inference parity."""
import importlib.util
import json
import math

import numpy as np
import pytest
from src.features.matching import (PairFeatureExtractor, BASE_NAMES, FAMILY_NAMES,
                                  address_parts, baseline_features, core_words)
from src.normalize import normalize, DIACRITIC_FOLDING_ENABLED


def query(eid='q', name='Acme Private Limited', address='0012 Main Rd, Suite 04, NY 10001'):
    return {'entity_id': eid, 'business_name': name, 'business_address': address, 'country': 'US'}


def pair(name='Acme Ltd', address='12 Main Road, Suite 4, NY 10001'):
    return {'source1_entity_id': 'q', 'target_entity_id': 't', 'target_name': name,
            'target_address': address, 'target_country': 'US', 'target_source': 's2'}


def legacy_module():
    spec = importlib.util.spec_from_file_location('legacy', 'scripts/22_missing_address_ablation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('qa,ta', [('', ''), ('', '12 main'), ('12 main', ''),
                                  ('10 main suite 3', '12 main suite 4'), ('rue école', 'rue école')])
def test_baseline_matches_legacy(qa, ta):
    args = ('acme', qa, 'us', 'acme inc', ta, 'us')
    np.testing.assert_allclose(baseline_features(*args), legacy_module().corrected_features(*args), equal_nan=True)
    assert len(baseline_features(*args)) == len(BASE_NAMES) == 16


def test_train_only_df_and_save_load_batch_parity(tmp_path):
    q = query()
    ex = PairFeatureExtractor(FAMILY_NAMES).fit([q, query('q2', 'Common Acme')])
    before = dict(ex.name_df)
    pairs = [pair(), pair('Heldout Unicorn'), pair(address='')]
    whole = ex.transform(pairs, {'q': q})
    assert ex.name_df == before and 'heldout' not in ex.name_df
    path = tmp_path / 'extractor.json'
    ex.save(path)
    restored = PairFeatureExtractor.load(path)
    batches = np.vstack([restored.transform([p], {'q': q}) for p in pairs])
    np.testing.assert_allclose(whole, batches, equal_nan=True)
    assert restored.feature_names == ex.feature_names
    assert restored.document_count == 2
    assert restored.transform([], {'q': q}).shape == (0, len(ex.feature_names))
    assert q['business_address'].startswith('0012')
    assert DIACRITIC_FOLDING_ENABLED is False


def test_schema_tampering_rejected(tmp_path):
    ex = PairFeatureExtractor(['name']).fit([query()])
    path = tmp_path / 'extractor.json'
    ex.save(path)
    payload = json.loads(path.read_text())
    payload['feature_names'].reverse()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='schema'):
        PairFeatureExtractor.load(path)


def test_rarity_distinguishes_generic_overlap():
    rows = [query(str(i), f'Common Business {i}') for i in range(20)] + [query('q', 'Common Rarebrand')]
    ex = PairFeatureExtractor(['name']).fit(rows)
    a = ex.transform([pair('Common Other'), pair('Rarebrand Other')], {'q': rows[-1]})
    col = ex.feature_names.index('name_idf_jaccard')
    assert a[1, col] > a[0, col]


def test_address_components_missing_is_not_conflict():
    ex = PairFeatureExtractor(['address']).fit([query()])
    arr = ex.transform([pair(), pair(address=''), pair(address='14 Main Road, Suite 5, NY 10002')], {'q': query()})
    cols = {n: i for i, n in enumerate(ex.feature_names)}
    for component in ('postcode', 'building', 'unit'):
        assert arr[0, cols[component + '_equal']] == 1
        assert math.isnan(arr[1, cols[component + '_conflict']])
        assert arr[1, cols[component + '_both_known']] == 0
        assert arr[2, cols[component + '_conflict']] == 1


@pytest.mark.parametrize('address', ['12-14 Main Rd', '12 - 14 Main Rd', '12/14 Main Rd', 'Main Rd'])
def test_ambiguous_house_number_unknown(address):
    assert address_parts(address, 'us')[1] is None


def test_country_specific_postcodes():
    assert address_parts('12 Main, CA 01234-5678', 'us')[0] == '01234'
    assert address_parts('12 Road Delhi 110001', 'india')[0] == '110001'
    assert address_parts('12 Rue Victor 75001 Paris', 'france')[0] == '75001'
    assert address_parts('12345', 'us')[0] is None
    assert address_parts('12 Road 110001 110002', 'india')[0] is None
    assert address_parts('12 Road Delhi 110001', 'unknown')[0] is None


def test_legal_core_and_acronym():
    assert core_words('private bank limited') == ['private', 'bank']
    ex = PairFeatureExtractor(['name']).fit([query(name='International Business Machines LLC')])
    values = ex.transform([pair('IBM Inc')], {'q': query(name='International Business Machines LLC')})
    assert values[0, ex.feature_names.index('name_acronym_match')] == 1


def test_alternate_views_do_not_change_originals():
    q = query(name='École', address='12 Main')
    ex = PairFeatureExtractor(['views']).fit([q])
    values = ex.transform([pair('Ecole', '12 Main')], {'q': q})
    assert values[0, ex.feature_names.index('name_romanized_token_sort')] == 1
    assert values[0, ex.feature_names.index('name_exact')] == 0
    indic = query(name='भारत')
    values = ex.transform([pair('Bharat')], {'q': indic})
    assert values[0, ex.feature_names.index('name_romanization_changed')] == 1
    assert values[0, ex.feature_names.index('name_romanized_token_sort')] > values[0, ex.feature_names.index('name_token_sort')]
    assert normalize('École') == 'école'


def test_fit_guards():
    with pytest.raises(ValueError, match='Duplicate'):
        PairFeatureExtractor().fit([query(), query()])
    with pytest.raises(ValueError, match='empty'):
        PairFeatureExtractor().fit([])
    with pytest.raises(ValueError, match='Fit'):
        PairFeatureExtractor().transform([pair()], {'q': query()})
    with pytest.raises(ValueError, match='family'):
        PairFeatureExtractor(['wrong'])


def test_transliteration_preserves_indic_vowel_signs():
    from anyascii import anyascii
    q = query(name='भारत')
    ex = PairFeatureExtractor(['views']).fit([q])
    target = normalize(anyascii(q['business_name']))
    values = ex.transform([pair(target)], {'q': q})
    assert values[0, ex.feature_names.index('name_romanized_token_sort')] == 1


def script_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_rejects_overlap_and_changed_pool(tmp_path):
    ablation = script_module('ablation', 'scripts/24_feature_ablation.py')
    rows = [query(str(i)) for i in range(3)]
    sample = tmp_path / 'sample.jsonl'
    sample.write_text('\n'.join(json.dumps(r) for r in rows))
    pairs = tmp_path / 'pairs.jsonl'
    p = pair()
    p['source1_entity_id'] = '0'
    pairs.write_text(json.dumps(p) + '\n')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['0'], 'calibration': ['0'], 'validation': ['2']}))
    with pytest.raises(ValueError, match='disjoint'):
        ablation.load_inputs(sample, pairs, split)
    split.write_text(json.dumps({'train': ['0'], 'calibration': ['1'], 'validation': ['2']}))
    allowed = tmp_path / 'allowed.tsv'
    allowed.write_text('source1_entity_id\ttarget_entity_id\n0\tmissing\n')
    with pytest.raises(ValueError, match='frozen pool'):
        ablation.load_inputs(sample, pairs, split, allowed)


def test_inference_uses_saved_extractor_and_threshold(tmp_path):
    import csv
    import hashlib
    from xgboost import XGBClassifier
    from src.features.matching import VERSION
    inference = script_module('inference', 'scripts/25_predict_features.py')
    q = query()
    pairs = [pair(), pair('Different Company', '85 Other Rd, NY 10003')] * 10
    ex = PairFeatureExtractor(FAMILY_NAMES).fit([q])
    x = ex.transform(pairs, {'q': q})
    model = XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(x, [1, 0] * 10)
    model.save_model(tmp_path / 'model.json')
    ex.save(tmp_path / 'extractor.json')
    meta = {'feature_version': VERSION, 'feature_names': ex.feature_names, 'threshold': .5,
            'model_sha256': hashlib.sha256((tmp_path / 'model.json').read_bytes()).hexdigest(),
            'extractor_sha256': hashlib.sha256((tmp_path / 'extractor.json').read_bytes()).hexdigest()}
    (tmp_path / 'bundle.json').write_text(json.dumps(meta))
    sample_path, pairs_path = tmp_path / 'sample.jsonl', tmp_path / 'pairs.jsonl'
    sample_path.write_text(json.dumps(q) + '\n')
    pairs_path.write_text('\n'.join(json.dumps(p) for p in pairs))
    output = tmp_path / 'scores.tsv'
    inference.score_batches(tmp_path, sample_path, pairs_path, output, batch_size=3)
    with output.open() as stream:
        scored = list(csv.DictReader(stream, delimiter='\t'))
    expected = model.predict_proba(x)[:, 1]
    np.testing.assert_allclose([float(r['score']) for r in scored], expected)
    assert [int(r['selected']) for r in scored] == (expected >= .5).astype(int).tolist()
    with pytest.raises(FileExistsError):
        inference.score_batches(tmp_path, sample_path, pairs_path, output)
    (tmp_path / 'extractor.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        inference.score_batches(tmp_path, sample_path, pairs_path, tmp_path / 'bad.tsv')
    assert not (tmp_path / 'bad.tsv').exists()
