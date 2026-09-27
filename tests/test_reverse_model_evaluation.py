"""Behavioral regression tests: pool integrity, missingness, leakage and inference."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.features import competition
from src.blocking.union import pair_key, union_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
runner = importlib.import_module('28_reverse_model_evaluation')


def test_unknown_is_nan_routes_are_explicit_and_ties_survive():
    rows = [dict(source1_entity_id='q', forward_score=.4, forward_rank=3,
                 reverse_rank_censored=True, retrieval_routes=['forward_structured']),
            dict(source1_entity_id='q', reverse_score=.7, reverse_rank=1,
                 reverse_gap_to_best=0, reverse_margin_to_best_other=0,
                 reverse_best_s1_id='other', reverse_rank_censored=False,
                 retrieval_routes=['reverse_sparse']),
            dict(source1_entity_id='q', reverse_score=.4, reverse_rank=3,
                 reverse_gap_to_best=.3, reverse_margin_to_best_other=-.3,
                 retrieval_routes=['forward_structured'])]
    before = copy.deepcopy(rows)
    values = competition.transform(rows)
    assert values.shape == (3, 12)
    assert np.isnan(values[0, 2:8]).all()
    assert values[0, 1] == 3 and values[0, 8] == 1
    assert np.isnan(values[1, :2]).all()
    assert values[1, 3] == 1 and values[1, 7] == 0 and values[1, 11] == 0
    assert values[2, 7] < 0 and values[2, 10] == 0  # Known rank is not route membership.
    assert rows == before
    assert competition.transform([]).shape == (0, 12)


@pytest.mark.parametrize('extra', [dict(reverse_rank=0), dict(reverse_rank=1.5),
                                  dict(reverse_score=float('inf')),
                                  dict(reverse_rank_censored=True, reverse_score=0),
                                  dict(retrieval_routes=['unknown'])])
def test_bad_competition_metadata_rejected(extra):
    with pytest.raises(ValueError):
        competition.transform([dict(source1_entity_id='q', **extra)])


def fixture_rows():
    queries, forward = {}, []
    for n in range(9):
        owner = f'S1-{n}'
        queries[owner] = dict(entity_id=owner, business_name=f'Acme {n}',
                              business_address=f'{n+1} Main Street', country='US',
                              matches=[f'S2-{n}', f'S3-absent{n}'])
        for true in (True, False):
            forward.append(dict(source1_entity_id=owner, target_entity_id=f'S2-{n}' if true else f'S2-false{n}',
                                target_source='s2', target_name=f'Acme {n}' if true else 'Unrelated Shop',
                                target_address=f'{n+1} Main Street' if true else '', target_country='US',
                                blocking_score=.9 if true else .2, blocking_rank=1 if true else 2))
    split = dict(train=list(queries)[:3], calibration=list(queries)[3:6], validation=list(queries)[6:])
    return queries, forward, split


def union_fixture(tmp_path):
    queries, forward, split = fixture_rows()
    hashes = {name: name + '-hash' for name in ('sample', 'pairs', 'split', 'allowed')}
    contexts = {}
    for row in forward:
        owner = row['source1_entity_id']
        contexts[row['target_entity_id']] = dict(best_s1_id=owner, best_score=.8, second_score=.2,
                                                 top_owners=[dict(source1_entity_id=owner, score=.8, rank=1)])
    reverse = dict(source1_entity_id='S1-0', target_entity_id='S3-new', target_source='s3',
                   target_name='Novel', target_address='', target_country='US', reverse_rank=1, reverse_score=.7)
    contexts['S3-new'] = dict(best_s1_id='S1-0', best_score=.7, second_score=.2,
                             top_owners=[dict(source1_entity_id='S1-0', score=.7, rank=1)])
    report = dict(complete=True, reverse_manifest=dict(complete=True, benchmark_only=False, k=8,
                  score_floor_strict=.35, sample_sha256=hashes['sample'], forward_pairs_sha256=hashes['allowed']),
                  inputs=dict(sample=hashes['sample'], split=hashes['split'], forward_pairs=hashes['allowed'],
                              forward_cache=hashes['pairs']), unions={})
    for k in (1, 3, 8):
        rows = list(union_records(forward, [reverse], contexts, k))
        path = tmp_path / f'union_k{k}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        report['unions'][str(k)] = dict(path=path.name, sha256=runner.sha(path))
    (tmp_path / 'report.json').write_text(json.dumps(report))
    return queries, forward, hashes, report


def test_completed_unions_preserve_pool_without_truth_injection(tmp_path):
    queries, forward, hashes, _ = union_fixture(tmp_path)
    unions, _ = runner.completed_unions(tmp_path, hashes, forward, queries)
    assert len(unions[8]) == len(forward) + 1
    assert all(not r['target_entity_id'].startswith('S3-absent') for r in unions[8])
    before = copy.deepcopy(unions)
    for row in queries.values():
        row['matches'] = ['ARBITRARY-LABEL']
    after, _ = runner.completed_unions(tmp_path, hashes, forward, queries)
    assert after == before


@pytest.mark.parametrize('corruption', ['partial', 'benchmark', 'hash', 'forward_rank', 'lost_edge', 'input_hash'])
def test_invalid_unions_fail_closed(tmp_path, corruption):
    queries, forward, hashes, report = union_fixture(tmp_path)
    if corruption == 'partial':
        report['complete'] = False
    elif corruption == 'benchmark':
        report['reverse_manifest']['benchmark_only'] = True
    elif corruption == 'input_hash':
        report['inputs']['split'] = 'wrong'
    else:
        path = tmp_path / 'union_k3.jsonl'
        rows = runner.read_rows(path)
        if corruption == 'forward_rank':
            rows[0]['forward_rank'] = 999
        elif corruption == 'lost_edge':
            rows.pop(0)
        else:
            rows[0]['target_name'] = 'changed'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        if corruption != 'hash':
            report['unions']['3']['sha256'] = runner.sha(path)
    (tmp_path / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runner.completed_unions(tmp_path, hashes, forward, queries)


def test_training_calibration_ignore_development_labels_and_bundle_roundtrip(tmp_path):
    queries, rows, split = fixture_rows()
    for row in rows:
        row.update(forward_score=row['blocking_score'], forward_rank=row['blocking_rank'],
                   reverse_rank_censored=True, retrieval_routes=['forward_structured'])
    truth = {i: set(r['matches']) for i, r in queries.items()}
    params = dict(n_estimators=4, max_depth=2, n_jobs=1, random_state=2026,
                  tree_method='hist', objective='binary:logistic')
    base = runner.candidate_sets(rows)
    first = runner.train_run(tmp_path/'first', rows, queries, split, truth, params, True, {}, {}, base)
    changed = {i: (set() if i in split['validation'] else t) for i, t in truth.items()}
    second = runner.train_run(tmp_path/'second', rows, queries, split, changed, params, True, {}, {}, base)
    assert first['model_sha256'] == second['model_sha256']
    assert first['threshold'] == second['threshold']
    assert first['calibration'] == second['calibration']
    assert first['candidate_metrics']['validation']['candidate_pair_recall'] == .5
    assert first['development'] != second['development']
    sample = tmp_path / 'sample.jsonl'
    sample.write_text(''.join(json.dumps(r) + '\n' for r in queries.values()))
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    args = SimpleNamespace(bundle=tmp_path/'first', sample=sample, pairs=pairs, output=tmp_path/'scored.tsv')
    runner.score(args)
    saved = pd.read_csv(tmp_path/'first'/'scores.tsv', sep='\t')
    scored = pd.read_csv(args.output, sep='\t')
    np.testing.assert_allclose(saved.score, scored.score, rtol=1e-6)
    assert scored.selected.tolist() == saved.selected.astype(int).tolist()
    # Artifact tampering cannot silently change inference.
    (tmp_path/'first'/'extractor.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        runner.load_bundle(tmp_path/'first')


def test_complete_comparison_retrains_all_prespecified_controls(tmp_path):
    union_dir = tmp_path / 'unions'
    union_dir.mkdir()
    queries, forward, _, report = union_fixture(union_dir)
    _, _, split = fixture_rows()
    sample = tmp_path / 'sample.jsonl'
    sample.write_text(''.join(json.dumps(r) + '\n' for r in queries.values()))
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in forward))
    splits = tmp_path / 'split.json'
    splits.write_text(json.dumps(split))
    allowed = tmp_path / 'allowed.tsv'
    runner.frame_for(forward).to_csv(allowed, sep='\t', index=False)
    baseline = tmp_path / 'baseline'
    args = SimpleNamespace(sample=sample, pairs=pairs, split=splits, allowed=allowed, output=baseline)
    runner.control.run(args)
    hashes = runner.input_hashes(args)
    report['inputs'] = dict(sample=hashes['sample'], split=hashes['split'],
                            forward_pairs=hashes['allowed'], forward_cache=hashes['pairs'])
    report['reverse_manifest'].update(sample_sha256=hashes['sample'], forward_pairs_sha256=hashes['allowed'])
    report['seconds'] = .01
    (union_dir / 'report.json').write_text(json.dumps(report))
    args.baseline = baseline
    args.unions = union_dir
    args.output = tmp_path / 'comparison'
    runner.compare(args)
    result = json.loads((args.output/'report.json').read_text())
    assert result['complete']
    assert len(result['runs']) == 7
    assert result['runs']['forward47+competition']['candidate_count'] == 18
    assert result['runs']['union_k8_47+competition']['candidate_count'] == 19
    assert result['runs']['union_k1_47']['feature_count'] == 47
    assert result['runs']['union_k3_47+competition']['feature_count'] == 59
    eligible = {'forward47': result['forward47'], **result['runs']}
    best = max(eligible, key=lambda k: (eligible[k]['calibration']['macro_F0.5'], k))
    assert result['selected_by_calibration'] == best


@pytest.fixture
def validation_case(tmp_path):
    union_dir = tmp_path/'unions'
    union_dir.mkdir()
    queries, forward, _, report = union_fixture(union_dir)
    _, _, split = fixture_rows()
    sample, pairs, splits, allowed = [tmp_path/n for n in ('sample.jsonl','pairs.jsonl','split.json','allowed.tsv')]
    sample.write_text(''.join(json.dumps(r)+'\n' for r in queries.values()))
    pairs.write_text(''.join(json.dumps(r)+'\n' for r in forward))
    splits.write_text(json.dumps(split))
    runner.frame_for(forward).to_csv(allowed, sep='\t', index=False)
    args = SimpleNamespace(sample=sample,pairs=pairs,split=splits,allowed=allowed,output=tmp_path/'baseline')
    runner.control.run(args)
    args.baseline=args.output
    args.output=tmp_path/'validation-result'
    args.unions=union_dir
    args.manifest=tmp_path/'manifest.json'
    subset_hashes=runner.validation_subset_hashes(args, split, forward)
    report['inputs']=dict(sample=subset_hashes['sample'], split=subset_hashes['split'],
                         forward_pairs=subset_hashes['allowed'],forward_cache=subset_hashes['pairs'])
    report['reverse_manifest'].update(emitted_s1_count=3,sample_sha256=subset_hashes['sample'],
                                      forward_pairs_sha256=subset_hashes['allowed'])
    args.manifest.write_text(json.dumps(report['reverse_manifest']))
    report['inputs']['reverse_manifest']=runner.sha(args.manifest)
    for k in (1,3,8):
        path=union_dir/f'union_k{k}.jsonl'
        rows=runner.read_rows(path)
        extra=copy.deepcopy(next(r for r in rows if r['target_entity_id']=='S3-new'))
        extra.update(source1_entity_id='S1-6',reverse_best_s1_id='S1-6')
        rows=[r for r in rows if r['source1_entity_id'] in split['validation']]+[extra]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report['unions'][str(k)]['sha256']=runner.sha(path)
    (union_dir/'report.json').write_text(json.dumps(report))
    return args,queries,forward,split


def test_validation_only_never_trains_or_recalibrates(validation_case, monkeypatch):
    args,queries,forward,split=validation_case
    def forbidden(*a,**kw):
        raise AssertionError('Validation-only must not train or select a threshold')
    monkeypatch.setattr(runner.XGBClassifier,'fit',forbidden)
    monkeypatch.setattr(runner,'select_threshold',forbidden)
    runner.validation_only(args)
    report=json.loads((args.output/'report.json').read_text())
    assert report['complete'] and report['scope']['mode']=='validation-only'
    assert report['scope']['validation_s1_count']==3
    assert not report['scope']['full_split_comparison']
    assert not report['scope']['model_retrained']
    assert not report['scope']['threshold_reselected']
    assert len(report['runs'])==8
    assert report['candidate_metrics']['forward']['candidate_count']==6
    assert report['candidate_metrics']['union_k8']['candidate_count']==7
    for path in args.output.glob('*.tsv'):
        frame=pd.read_csv(path,sep='\t')
        assert set(frame.source1_entity_id)<=set(split['validation'])
    # The same subset cannot masquerade as a full-split comparison.
    with pytest.raises(ValueError,match='input hashes'):
        runner.completed_unions(args.unions,runner.input_hashes(args),forward,queries)


@pytest.mark.parametrize('corruption',['subset_hash','manifest','foreign_owner'])
def test_validation_only_rejects_bad_provenance(validation_case,corruption):
    args,_,_,_=validation_case
    path=args.unions/'report.json'
    report=json.loads(path.read_text())
    if corruption=='subset_hash':
        report['inputs']['forward_cache']='wrong'
    elif corruption=='manifest':
        args.manifest.write_text(args.manifest.read_text()+'\n')
    else:
        union=args.unions/'union_k8.jsonl'
        rows=runner.read_rows(union)
        extra=copy.deepcopy(rows[-1]);extra['source1_entity_id']='S1-0'
        rows.append(extra)
        union.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report['unions']['8']['sha256']=runner.sha(union)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runner.validation_only(args)
    assert not args.output.exists()
