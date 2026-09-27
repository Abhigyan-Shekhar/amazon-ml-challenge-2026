import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from src.blocking.training_name import build_alias_map, retrieve_name_aliases, verify_alias_map


def row(entity_id, name):
    return {'entity_id': entity_id, 'business_name': name, 'business_address': '', 'country': 'US'}


def test_alias_requires_independent_linked_entities_and_keeps_evidence():
    s1 = [row('a1', 'Acme Pvt Ltd'), row('a2', 'Acme Pvt Ltd'), row('lonely', 'One Name')]
    s2 = [row('b1', 'ACME Private Limited'), row('b2', 'ACME Private Limited'), row('only', 'Other Name')]
    mapping = build_alias_map(s1, s2, [], {'a1': {'b1'}, 'a2': {'b2'}, 'lonely': {'only'}},
                              {'labels': 'abc'}, min_support=2)
    assert mapping['aliases']['acme pvt ltd'] == mapping['aliases']['acme private limited']
    assert 'one name' not in mapping['aliases']
    proof = mapping['evidence'][0]
    assert proof['training_s1_ids'] == ['a1', 'a2']
    assert proof['training_target_ids'] == ['b1', 'b2']
    assert proof['training_pairs'] == [['a1', 'b1'], ['a2', 'b2']]
    assert mapping['input_sha256'] == {'labels': 'abc'}
    assert len(mapping['map_sha256']) == 64
    assert verify_alias_map(mapping)
    altered = dict(mapping, aliases={})
    assert not verify_alias_map(altered)


def test_no_single_link_can_create_a_map_and_route_preserves_original_name():
    s1 = [row('a', 'München Bakery')]
    s2 = [row('b', 'Munchen Bakery')]
    mapping = build_alias_map(s1, s2, [], {'a': {'b'}}, {'train': 'hash'})
    assert mapping['aliases'] == {}
    rows = retrieve_name_aliases(s1, s2, mapping)
    assert rows == []


def test_alias_retrieval_is_distinct_route_with_score_rank_metadata():
    s1 = [row('a1', 'Acme Pvt Ltd'), row('a2', 'Acme Pvt Ltd')]
    s2 = [row('b1', 'ACME Private Limited'), row('b2', 'ACME Private Limited')]
    mapping = build_alias_map(s1, s2, [], {'a1': {'b1'}, 'a2': {'b2'}}, {'split': 'hash'})
    rows = retrieve_name_aliases(s1[:1], [dict(s2[0], target_source='s2')], mapping)
    assert len(rows) == 1
    assert rows[0]['target_name'] == 'ACME Private Limited'
    assert rows[0]['retrieval_routes'] == ['training_name_alias']
    assert rows[0]['name_alias_score'] > 0
    assert rows[0]['name_alias_rank'] == 1


def test_cli_map_uses_train_partition_only_and_records_hashes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('training_name_cli', root/'scripts'/'28_training_name_route.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    header = 'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
    s1 = tmp_path/'s1.tsv'; s2 = tmp_path/'s2.tsv'; s3 = tmp_path/'s3.tsv'
    s1.write_text(header+'a1\tAlpha Pvt Ltd\t\tUS\na2\tAlpha Pvt Ltd\t\tUS\ndev\tDev Alias\t\tUS\n')
    s2.write_text(header+'b1\tAlpha Private Limited\t\tUS\nb2\tAlpha Private Limited\t\tUS\n')
    s3.write_text(header)
    labels = tmp_path/'labels.tsv'
    labels.write_text('source1_entity_id\tmatched_entity_ids\na1\tb1\na2\tb2\ndev\tb1\n')
    split = tmp_path/'split.json'; split.write_text(json.dumps({'train':['a1','a2'],'development':['dev']}))
    output = tmp_path/'map.json'
    cli.build(SimpleNamespace(s1=s1,s2=s2,s3=s3,labels=labels,split=split,output=output,min_support=2))
    mapping = json.loads(output.read_text())
    assert mapping['training_s1_ids'] == ['a1','a2']
    assert mapping['training_partition_s1_ids'] == ['a1','a2']
    assert mapping['training_target_ids'] == ['b1','b2']
    assert all('dev' not in pair for pair in mapping['training_pairs'])
    assert mapping['input_sha256']['labels'] == cli.sha(labels)
    assert len(mapping['map_sha256']) == 64


def test_cli_retrieve_then_lossless_candidate_ablation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('training_name_cli_e2e', root/'scripts'/'28_training_name_route.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    header = 'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
    s1 = tmp_path/'s1.tsv'; s2 = tmp_path/'s2.tsv'; s3 = tmp_path/'s3.tsv'
    s1.write_text(header+'a1\tAlpha Pvt Ltd\t1 Main Rd\tUS\na2\tAlpha Pvt Ltd\t2 Main Rd\tUS\ndev\tAlpha Pvt Ltd\t3 Main Rd\tUS\n')
    s2.write_text(header+'b1\tAlpha Private Limited\t1 Main Rd\tUS\nb2\tAlpha Private Limited\t2 Main Rd\tUS\nbdev\tAlpha Private Limited\t3 Main Rd\tUS\n')
    s3.write_text(header)
    sample = tmp_path/'sample.jsonl'
    sample.write_text('\n'.join(json.dumps({'entity_id': i, 'business_name': 'Alpha Pvt Ltd',
                                             'business_address': '', 'country': 'US', 'matches': m})
                                   for i,m in [('a1',['b1']),('a2',['b2']),('dev',['bdev'])])+'\n')
    split = tmp_path/'split.json'; split.write_text(json.dumps({'train':['a1','a2'],'development':['dev']}))
    labels = tmp_path/'labels.tsv'
    labels.write_text('source1_entity_id\tmatched_entity_ids\na1\tb1\na2\tb2\ndev\tbdev\n')
    map_path = tmp_path/'map.json'
    cli.build(SimpleNamespace(s1=s1,s2=s2,s3=s3,labels=labels,split=split,output=map_path,min_support=2))
    route_dir = tmp_path/'route'
    cli.retrieve(SimpleNamespace(s1=s1,s2=s2,s3=s3,sample=sample,map=map_path,output=route_dir))
    assert json.loads((route_dir/'manifest.json').read_text())['candidate_edges'] == 9

    forward_pairs = tmp_path/'forward.tsv'
    forward_pairs.write_text('source1_entity_id\ttarget_entity_id\ttarget_source\n')
    forward_cache = tmp_path/'forward.jsonl'; forward_cache.write_text('')
    report_dir = tmp_path/'report'
    cli.evaluate_route(SimpleNamespace(sample=sample,split=split,labels=labels,forward_pairs=forward_pairs,
                                       forward_cache=forward_cache,name_route=route_dir,output=report_dir))
    report = json.loads((report_dir/'report.json').read_text())
    dev = report['splits']['development']
    assert dev['name_route_only']['candidate_pair_recall'] == 1.0
    assert dev['union']['candidate_pair_recall'] == 1.0
    assert dev['new_true_links'] == 1 and dev['lost_true_links'] == 0
