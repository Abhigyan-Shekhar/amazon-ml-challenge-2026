import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
from src.evaluate import entity_f05, evaluate, blocking_metrics
from src.split import grouped_split
from src.normalize import normalize, prepare
from src.blocking.char_tfidf import retrieve
from src.generate_submission import write_submission, validate_submission
from src.data import read_truth

@pytest.mark.parametrize('truth,predicted,expected', [({'a'}, {'a'}, 1), (set(), set(), 1), (set(), {'a'}, 0), ({'a'}, set(), 0), ({'a','b'}, {'a'}, 1.25/1.5), ({'a'}, {'a','b'}, 1.25/2.25)])
def test_f05(truth, predicted, expected):
    assert entity_f05(truth, predicted) == pytest.approx(expected)

def test_macro_all_empty():
    assert evaluate({'a': set(), 'b': {'z'}}, {})['macro_F0.5'] == 0.5

def test_blocking_complete():
    result = blocking_metrics({'a': {'x','y','z'}, 'b': set()}, {'a': {'x','y'}})
    assert result['candidate_pair_recall'] == pytest.approx(2/3)
    assert result['entity_complete_coverage'] == 0.5
    assert result['matched_entity_complete_coverage'] == 0

def test_split():
    a,b = grouped_split([str(i) for i in range(20)])
    assert not set(a)&set(b)
    assert len(a)+len(b)==20
    assert (a,b)==grouped_split([str(i) for i in range(20)])

def test_normalize():
    assert normalize('ÉCOLE—Paris, １２') == 'école paris 12'

def records(rows):
    return prepare(pd.DataFrame(rows, columns=['entity_id','business_name','business_address','country']))

def test_retrieval_and_output(tmp_path):
    sources = {'s1': records([('a','École Paris','12 Rue','France'), ('b','','','')]), 's2': records([('x','École Paris','12 Rue','France')]), 's3': records([('y','École Paris','12 Rue','')])}
    candidates, _ = retrieve(sources, k=1)
    assert set(candidates.target_entity_id)=={'x','y'}
    write_submission(tmp_path/'outputs', sources, candidates, {'a': {'x','y'}})
    matching = pd.read_csv(tmp_path/'outputs/matching_results.tsv', sep='\t', keep_default_na=False)
    assert matching.iloc[1].matched_entity_ids == ''
    with pytest.raises(ValueError):
        write_submission(tmp_path/'bad', sources, candidates, {'b': {'x'}})

def test_missing_labels_fail(tmp_path):
    sources = {'s1': records([('a','','',''),('b','','','')]), 's2': records([]), 's3': records([])}
    path = tmp_path/'labels.tsv'
    pd.DataFrame([('a','')], columns=['source1_entity_id','matched_entity_ids']).to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError, match='explicitly cover'):
        read_truth(path, sources)

def test_end_to_end_synthetic(tmp_path):
    config = {'seed': 2026}
    for part in ['train','test']:
        config[part] = {}
        for source in ['s1','s2','s3']:
            rows = [(f'{source}_{i}', f'Synthetic Business {i}', f'{i} Example Street', 'India' if i%2 else 'US') for i in range(20)]
            path = tmp_path/f'{part}_{source}.tsv'
            records(rows)[['entity_id','business_name','business_address','country']].to_csv(path, sep='\t', index=False)
            config[part][source] = str(path)
        if part == 'train':
            path = tmp_path/'labels.tsv'
            pd.DataFrame([(f's1_{i}', json.dumps([] if i%4==0 else [f's2_{i}', f's3_{i}'])) for i in range(20)], columns=['source1_entity_id','matched_entity_ids']).to_csv(path, sep='\t', index=False)
            config[part]['labels'] = str(path)
    config_path = tmp_path/'config.json'
    config['eda_output'] = str(tmp_path/'eda.json')
    config_path.write_text(json.dumps(config))
    subprocess.run([sys.executable, 'scripts/01_eda.py', '--config', str(config_path)], check=True)
    eda = json.loads((tmp_path/'eda.json').read_text())
    assert eda['labels']['zero'] == 5
    assert eda['labels']['many'] == 15
    assert eda['labels']['shared_target_count'] == 0
    subprocess.run([sys.executable, 'scripts/02_baseline.py', '--config', str(config_path), '--output-root', str(tmp_path/'runs'), '--experiment-id','synthetic'], check=True)
    run = tmp_path/'runs/synthetic'
    assert (run/'outputs/candidate_pairs.tsv').exists()
    manifest = json.loads((run/'final_manifest.json').read_text())
    assert manifest['model_name']=='char_tfidf_cosine'
    report = json.loads((run/'validation.json').read_text())
    assert not set(report['split']['calibration']) & set(report['split']['validation'])


def test_threshold_can_reject_perfect_cosine_singleton():
    from src.predict import select_threshold, predict
    candidates = pd.DataFrame([('001','002',1.0)], columns=['source1_entity_id','target_entity_id','score'])
    threshold, _ = select_threshold(candidates, {'001': set()}, ['001'])
    assert threshold > 1
    assert predict(candidates, threshold) == {}

def test_duplicate_output_match_rejected():
    sources = {'s1': records([('001','','','')]), 's2': records([('002','','','')]), 's3': records([])}
    candidates = pd.DataFrame([('001','002')],columns=['source1_entity_id','target_entity_id'])
    matching = pd.DataFrame([('001','["002", "002"]')],columns=['source1_entity_id','matched_entity_ids'])
    with pytest.raises(ValueError,match='Duplicate match'):
        validate_submission(sources,candidates,matching)
