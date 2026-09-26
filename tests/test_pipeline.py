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

def test_missing_address_features_are_explicitly_missing():
    import importlib.util
    import math
    from src.features.fuzzy import fuzzy_features
    spec = importlib.util.spec_from_file_location('sample_experiment', 'scripts/12_sample_experiment.py')
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    absent = experiment.cheap_features('acme', '', 'us', 'acme', '12 main', 'us', neutral_missing_address=True)
    fuzzy_absent = fuzzy_features('acme', '', 'acme', '12 main', neutral_missing_address=True)
    present_mismatch = experiment.cheap_features('acme', '10 main', 'us', 'acme', '12 main', 'us', neutral_missing_address=True)
    assert absent[9] == 1 and present_mismatch[9] == 0
    assert all(math.isnan(absent[i]) for i in (1, 3, 4, 5, 8))
    assert all(math.isnan(fuzzy_absent[i]) for i in (1, 3, 5))
    assert all(not math.isnan(present_mismatch[i]) for i in (1, 3, 4, 5, 8))
    # The archived v1 checkpoint still gets its original feature schema.
    assert experiment.cheap_features('acme', '', 'us', 'acme', '12 main', 'us')[1] == 0
    assert fuzzy_features('acme', '', 'acme', '12 main')[1] == 0

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

def test_official_format_and_validator(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('official', 'utils/validate_submission.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    sources = {'s1': records([('S1-1','Acme','1 Main','US'),('S1-2','Other','2 Main','France')]), 's2': records([('S2-1','Acme','1 Main','US')]), 's3': records([])}
    for i in (1,2,3):
        sources[f's{i}'][['entity_id','business_name','business_address','country']].to_csv(tmp_path/f'test_source{i}.tsv',sep='\t',index=False)
    candidates = pd.DataFrame([('S1-1','S2-1')], columns=['source1_entity_id','target_entity_id'])
    write_submission(tmp_path/'out',sources,candidates,{'S1-1':{'S2-1'}})
    assert (tmp_path/'out/candidate_pairs.tsv').read_text() == 'source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n'
    errors,warnings=official.validate(str(tmp_path/'out/matching_results.tsv'),str(tmp_path/'out/candidate_pairs.tsv'),str(tmp_path),check_ids=True)
    assert errors == warnings == []


def test_blocking_keys_open_country():
    from src.blocking.keys import query_keys,target_keys
    q=query_keys('ecole paris limited','12 rue de paris',{'ecole':1,'paris':2,'limited':100})
    t=target_keys('paris ecole ltd','12 rue de paris')
    assert q&t


def test_sample_experiment_respects_requested_k(tmp_path):
    candidates=tmp_path/'candidates';candidates.mkdir()
    sample=[{'entity_id':f'S1-{i}','business_name':f'Acme {i}','business_address':f'{i} Main Street','country':'India' if i%2 else 'US','matches':[f'S2-{i}-0'] if i%4 else []} for i in range(20)]
    (candidates/'sample_s1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sample))
    with (candidates/'pairs.jsonl').open('w') as f:
        for q in sample:
            i=q['entity_id'].split('-')[1]
            for j in range(3):
                f.write(json.dumps({'source1_entity_id':q['entity_id'],'target_entity_id':f'S2-{i}-{j}','target_name':q['business_name'] if j==0 else f'Other Store {j}','target_address':q['business_address'] if j==0 else f'{j} Side Road','target_country':q['country'],'target_source':'s2','blocking_score':1-j*.2,'blocking_rank':j+1})+'\n')
    (candidates/'benchmark.json').write_text('{}')
    run=tmp_path/'run'
    subprocess.run([sys.executable,'scripts/12_sample_experiment.py','--candidates',str(candidates),'--output',str(run),'--ranking','blocking','--k','1'],check=True,capture_output=True,text=True)
    report=json.loads((run/'validation.json').read_text())
    assert report['final_k_per_source']==1
    final=pd.read_csv(run/'candidate_pairs_internal.tsv',sep='\t')
    assert final.groupby('source1_entity_id').size().max()==1


def test_full_streaming_fallback_matches_frozen_classifier(tmp_path):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    test_dir=tmp_path/'test';test_dir.mkdir();model_dir=tmp_path/'model';model_dir.mkdir()
    data={'s1':records([('S1-1','Acme LLC','12 Main Road','US'),('S1-2','Unique','99 Long Road','France')]),'s2':records([('S2-1','Acme Inc','12 Main Road','US')]),'s3':records([])}
    for i in (1,2,3):data[f's{i}'][['entity_id','business_name','business_address','country']].to_csv(test_dir/f'test_source{i}.tsv',sep='\t',index=False)
    rng=np.random.default_rng(2026);x=rng.random((50,10));y=(x[:,1]>.5).astype(int)
    model=make_pipeline(StandardScaler(),LogisticRegression()).fit(x,y)
    joblib.dump({'k_per_source':1,'cheap_model':model},model_dir/'lexical_model.joblib')
    report={'ranking':'blocking','blocking':{'exact_core_address':True},'results':{'cheap_logistic':{'threshold':.5,'validation':{'macro_F0.5':0},'slices':{}}},'git_commit':'fixture'}
    (model_dir/'validation.json').write_text(json.dumps(report))
    output=tmp_path/'out'
    subprocess.run([sys.executable,'scripts/15_full_exact_baseline.py','--test-dir',str(test_dir),'--model-dir',str(model_dir),'--output',str(output)],check=True,capture_output=True,text=True)
    final=pd.read_csv(output/'matching_results.tsv',sep='\t',keep_default_na=False)
    # Features for this known pair, in the documented frozen order.
    features=np.array([[1/3,1,0,1,1,0,1,1,1,0]])
    expected='S2-1' if model.predict_proba(features)[0,1]>=.5 else ''
    assert final.iloc[0].matched_entity_ids==expected
    assert final.iloc[1].matched_entity_ids==''
    candidates=pd.read_csv(output/'candidate_pairs.tsv',sep='\t',keep_default_na=False)
    assert candidates.iloc[0].candidate_entity_ids=='S2-1'
    from sklearn.ensemble import HistGradientBoostingClassifier
    tree_dir=tmp_path/'tree';tree_dir.mkdir()
    tree=HistGradientBoostingClassifier(max_iter=3,min_samples_leaf=2,early_stopping=False,random_state=2026).fit(x,y)
    joblib.dump({'model':tree,'threshold':.5},tree_dir/'model.joblib')
    (tree_dir/'validation.json').write_text(json.dumps({'cheap_only':True,'validation':{'macro_F0.5':0},'slices':{}}))
    report['blocking']['max_key_frequency']=10
    (model_dir/'validation.json').write_text(json.dumps(report))
    structured=tmp_path/'structured'
    subprocess.run([sys.executable,'scripts/18_full_structured.py','--test-dir',str(test_dir),'--model-dir',str(model_dir),'--tree-dir',str(tree_dir),'--max-key-frequency','10','--output',str(structured)],check=True,capture_output=True,text=True)
    actual=pd.read_csv(structured/'matching_results.tsv',sep='\t',keep_default_na=False)
    assert actual.iloc[0].matched_entity_ids==('S2-1' if tree.predict_proba(features)[0,1]>=.5 else '')
    assert actual.iloc[1].matched_entity_ids==''
    from src.features.fuzzy import fuzzy_features
    augmented=np.column_stack([x,rng.random((len(x),6))])
    fuzzy_tree=HistGradientBoostingClassifier(max_iter=3,min_samples_leaf=2,early_stopping=False,random_state=2026).fit(augmented,y)
    joblib.dump({'model':fuzzy_tree,'threshold':.5,'uses_fuzzy':True},tree_dir/'model.joblib')
    fuzzy_out=tmp_path/'fuzzy'
    subprocess.run([sys.executable,'scripts/18_full_structured.py','--test-dir',str(test_dir),'--model-dir',str(model_dir),'--tree-dir',str(tree_dir),'--max-key-frequency','10','--output',str(fuzzy_out)],check=True,capture_output=True,text=True)
    augmented_expected=np.column_stack([features,[fuzzy_features('acme llc','12 main road','acme inc','12 main road')]])
    actual=pd.read_csv(fuzzy_out/'matching_results.tsv',sep='\t',keep_default_na=False)
    assert actual.iloc[0].matched_entity_ids==('S2-1' if fuzzy_tree.predict_proba(augmented_expected)[0,1]>=.5 else '')
    assert actual.iloc[1].matched_entity_ids==''


def test_fuzzy_features_blank_address_not_exact_match():
    from src.features.fuzzy import fuzzy_features
    values=fuzzy_features('acme','', 'acme','')
    assert values[0]==values[2]==values[4]==1
    assert values[1]==values[3]==values[5]==0


def test_submission_package_format(tmp_path):
    import zipfile
    test_dir = tmp_path / 'test_data'
    test_dir.mkdir()
    # Write mock test source files
    (test_dir / 'test_source1.tsv').write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAcme\t1 Main\tUS\nS1-2\tOther\t2 Main\tFrance\n")
    (test_dir / 'test_source2.tsv').write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tAcme\t1 Main\tUS\n")
    (test_dir / 'test_source3.tsv').write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
    
    run_dir = tmp_path / 'mock_run'
    run_dir.mkdir()
    (run_dir / 'matching_results.tsv').write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
    (run_dir / 'candidate_pairs.tsv').write_text("source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
    (run_dir / 'result.json').write_text(json.dumps({'s1_rows': 2, 'candidates': 1, 'matches': 1}))
    
    # Mock model and manifest
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    model_path = run_dir / 'model_checkpoint.joblib'
    joblib.dump({'cheap_model': pipe}, model_path)
    
    manifest = {
        'model_name': 'cheap_logistic',
        'checkpoint_path': str(model_path.resolve()),
        'feature_config': ['f1', 'f2'],
        'threshold': 0.585
    }
    (run_dir / 'final_manifest.json').write_text(json.dumps(manifest))
    
    # Run packaging script with team name BlackList
    cmd = [
        sys.executable, 'scripts/17_package_baseline.py',
        '--run', str(run_dir),
        '--test-dir', str(test_dir),
        '--team-name', 'BlackList'
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    archive_path = run_dir.parent / 'BlackList_submission.zip'
    assert archive_path.is_file(), "Expected BlackList_submission.zip to exist"
    
    with zipfile.ZipFile(archive_path, 'r') as z:
        names = z.namelist()
        # Verify NO backslashes anywhere in the archive entry paths
        assert all('\\' not in name for name in names), f"Found backslash in zip paths: {[n for n in names if '\\' in n]}"
        
        # Verify top-level structure: strictly output/, code/, Documentation_template.md
        top_level = {name.split('/')[0] for name in names}
        assert top_level <= {'output', 'code', 'Documentation_template.md'}, f"Unexpected root entries: {top_level}"
        
        # Verify required outputs
        assert 'output/matching_results.tsv' in names
        assert 'output/candidate_pairs.tsv' in names
        assert 'Documentation_template.md' in names
        
        # Verify code/business_entity_resolution/ structure
        assert 'code/business_entity_resolution/README.md' in names
        assert 'code/business_entity_resolution/requirements.txt' in names
        assert any(n.startswith('code/business_entity_resolution/src/') for n in names)

