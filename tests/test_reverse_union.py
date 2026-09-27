import pytest
from src.blocking.union import union_records


def row(owner='a',target='t',**fields):
    return dict(source1_entity_id=owner,target_entity_id=target,target_source='s2',target_name='name',
                target_address='address',target_country='US',**fields)


def context():
    return {'t':dict(best_s1_id='b',best_score=.9,second_score=.8,top_owners=[
        dict(source1_entity_id='b',score=.9,rank=1),dict(source1_entity_id='a',score=.8,rank=2)])}


def test_lossless_nested_union_and_score_provenance():
    forward=[row(blocking_score=.6,blocking_rank=4),row(owner='c',blocking_score=.5,blocking_rank=5)]
    reverse=[row(reverse_score=.8,reverse_rank=2),row(owner='b',reverse_score=.9,reverse_rank=1)]
    one=list(union_records(forward,reverse,context(),1))
    three=list(union_records(forward,reverse,context(),3))
    assert len(one)==len(three)==3
    assert one[0]['retrieval_routes']==['forward_structured']
    assert three[0]['retrieval_routes']==['forward_structured','reverse_sparse']
    assert three[0]['blocking_score']==.6 and three[0]['forward_rank']==4
    assert three[0]['reverse_rank']==2 and three[0]['reverse_margin_to_best_other']==pytest.approx(-.1)
    assert three[1]['blocking_score'] is None
    assert three[2]['reverse_rank_censored'] and three[2]['reverse_score'] is None
    assert 'retrieval_routes' not in forward[0]  # Do not mutate caller's caches.


def test_reject_duplicate_mismatch_and_missing_context():
    with pytest.raises(ValueError,match='Duplicate'):
        list(union_records([row(),row()],[],context(),1))
    bad=row(reverse_rank=1);bad['target_name']='changed'
    with pytest.raises(ValueError,match='mismatch'):
        list(union_records([row()],[bad],context(),1))
    with pytest.raises(ValueError,match='Missing'):
        list(union_records([row()],[],{},1))
