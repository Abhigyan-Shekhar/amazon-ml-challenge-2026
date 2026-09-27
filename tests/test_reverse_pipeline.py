"""Exercise full-scan, interrupted resume, and union evaluation on real tiny TSVs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]


def load(name, script):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cli=load('reverse_cli','26_reverse_retrieval.py')
    union=load('union_cli','27_reverse_union.py')
    s1=tmp_path/'s1.tsv';s2=tmp_path/'s2.tsv';s3=tmp_path/'s3.tsv'
    header='entity_id\tbusiness_name\tbusiness_address\tcountry\n'
    s1.write_text(header+'a\tAlpha Bakery\t12 Main Road\tUS\nb\tBeta Cafe\t99 Other Street\tUS\nc\tGamma Shop\t9 Oak Drive\tUS\n')
    s2.write_text(header+'t\tAlpha Bakery\t12 Main Road\tUS\nu\tBeta Cafe\t99 Other Street\tUS\n')
    s3.write_text(header+'v\tAlpha Bakery\t12 Main Road\tUS\nw\tGamma Shop\t9 Oak Drive\tUS\n')
    monkeypatch.setattr(cli, 'vectorizers', lambda: (TfidfVectorizer(), TfidfVectorizer()))
    index=tmp_path/'index'
    cli.build(SimpleNamespace(s1=s1,output=index,name_weight=.5,memory_gib=8,budget_seconds=60))
    sample=tmp_path/'sample.jsonl'
    sample.write_text(json.dumps(dict(entity_id='a',business_name='Alpha Bakery',business_address='12 Main Road',country='US',matches=['t','v']))+'\n')
    forward=tmp_path/'forward.tsv';forward.write_text('source1_entity_id\ttarget_entity_id\ttarget_source\na\tt\ts2\n')
    cache=tmp_path/'cache.jsonl';cache.write_text(json.dumps(dict(source1_entity_id='a',target_entity_id='t',target_source='s2',target_name='Alpha Bakery',target_address='12 Main Road',target_country='US',blocking_score=.9,blocking_rank=1))+'\n')
    split=tmp_path/'split.json';split.write_text(json.dumps(dict(validation=['a'])))
    args=SimpleNamespace(index=index,s2=s2,s3=s3,sample=sample,forward_pairs=forward,k=8,floor=.35,
         threads=1,batch_size=1,witness_stride=2,max_targets_per_source=0,output=tmp_path/'reverse',
         memory_gib=8,budget_seconds=60,resume=False)
    ua=SimpleNamespace(reverse_dir=args.output,sample=sample,split=split,forward_pairs=forward,
                       forward_cache=cache,output=tmp_path/'union',ks=[1,3,8])
    return cli,union,args,ua


def test_full_scan_union_evaluation(setup):
    cli,union,args,ua=setup
    cli.retrieve(args);union.run(ua)
    report=json.loads((ua.output/'report.json').read_text())
    assert report['baseline']['all']['candidate_pair_recall']==.5
    assert report['unions']['1']['metrics']['all']['candidate_pair_recall']==1
    assert report['unions']['1']['metrics']['all']['new_true_links']==1
    assert report['unions']['8']['metrics']['all']['lost_true_links']==0
    assert report['reverse_manifest']['scanned_by_source']=={'s2':2,'s3':2}


def test_resume_rolls_back_uncommitted_batch_and_matches_clean_run(setup,monkeypatch):
    cli,union,args,ua=setup
    real_guard=cli.guard
    calls=0
    def interrupt(*a):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated interruption')
        return real_guard(*a)
    monkeypatch.setattr(cli,'guard',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        cli.retrieve(args)
    assert json.loads((args.output/'checkpoint.json').read_text())['scanned']==1
    monkeypatch.setattr(cli,'guard',real_guard)
    args.resume=True;cli.retrieve(args)
    resumed=args.output
    args.resume=False;args.output=args.output.parent/'clean';cli.retrieve(args)
    for name in ('reverse_pairs.jsonl','target_context.jsonl'):
        assert (resumed/name).read_bytes()==(args.output/name).read_bytes()
    assert json.loads((resumed/'manifest.json').read_text())['scanned_targets']==4


def test_prefix_benchmark_rejected_for_evaluation(setup):
    cli,union,args,ua=setup
    args.max_targets_per_source=1;cli.retrieve(args)
    with pytest.raises(ValueError,match='partial/prefix'):
        union.run(ua)


def test_changed_input_rejected_on_resume(setup,monkeypatch):
    cli,union,args,ua=setup
    def interrupt(*a):raise RuntimeError('simulated interruption')
    monkeypatch.setattr(cli,'guard',interrupt)
    with pytest.raises(RuntimeError):cli.retrieve(args)
    args.resume=True;args.floor=.4
    with pytest.raises(ValueError,match='mismatch'):cli.retrieve(args)
