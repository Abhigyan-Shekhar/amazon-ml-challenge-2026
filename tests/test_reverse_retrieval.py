import numpy as np
import pytest
from scipy import sparse
from src.blocking.reverse import combine, retrieve_batch


def test_full_competitors_not_sample_only():
    # Sampled owner 0 looks plausible, but owner 1 is strictly better.
    full=sparse.csr_matrix(np.array([[.8,.2],[1.,0.],[0.,1.]],dtype=np.float32))
    target=sparse.csr_matrix(np.array([[1.,0.]],dtype=np.float32))
    result=retrieve_batch(target,full.T.tocsr(),k=1,floor=.1,
                          emit_indices=[0],emit_index_transposed=full[[0]].T.tocsr())
    assert result.hits==[]
    assert result.contexts[0]['best_owner_row']==1
    result=retrieve_batch(target,full.T.tocsr(),k=3,floor=.1,
                          emit_indices=[0],emit_index_transposed=full[[0]].T.tocsr())
    assert result.hits[0]['reverse_rank']==2
    assert result.hits[0]['reverse_margin_to_best_other']==pytest.approx(-.2)


def test_gate_exactly_matches_ungated_subset_and_batch_sizes():
    rng=np.random.default_rng(1)
    full=sparse.random(40,80,density=.15,random_state=rng,dtype=np.float32).tocsr()
    targets=sparse.random(17,80,density=.15,random_state=rng,dtype=np.float32).tocsr()
    emit=[2,6,11,39]
    all_hits=retrieve_batch(targets,full.T.tocsr(),k=8,floor=.1).hits
    expected=[h for h in all_hits if h['owner_row'] in emit]
    got=retrieve_batch(targets,full.T.tocsr(),k=8,floor=.1,emit_indices=emit,
                       emit_index_transposed=full[emit].T.tocsr()).hits
    assert got==expected
    pieces=[]
    for i in range(len(targets.indptr)-1):
        result=retrieve_batch(targets[i],full.T.tocsr(),k=8,floor=.1,
                              emit_indices=emit,emit_index_transposed=full[emit].T.tocsr())
        for h in result.hits:h['target_row']+=i
        pieces+=result.hits
    assert pieces==got


def test_boundary_ties_retained_and_nested():
    full=sparse.csr_matrix(np.array([[1.,0.]]*12+[[.5,0.]],dtype=np.float32))
    target=sparse.csr_matrix(np.array([[1.,0.]],dtype=np.float32))
    for k in [1,3,8]:
        result=retrieve_batch(target,full.T.tocsr(),k=k,floor=.1)
        assert {h['owner_row'] for h in result.hits}==set(range(12))
        assert all(h['reverse_rank']==1 for h in result.hits)
        assert all(h['reverse_margin_to_best_other']==0 for h in result.hits)


def test_floor_empty_and_forced_context():
    full=sparse.eye(3,format='csr',dtype=np.float32)
    target=sparse.csr_matrix(np.array([[0,.2,0],[0,0,0]],dtype=np.float32))
    r=retrieve_batch(target,full.T,k=1,floor=.3,emit_indices=[0],emit_index_transposed=full[[0]].T,force_rows=[0,1])
    assert r.hits==[] and r.targets_scored_globally==2
    assert r.contexts[0]['best_owner_row']==1 and r.contexts[1]['best_owner_row'] is None


def test_combined_score_missing_field_not_renormalized():
    n=sparse.csr_matrix([[1.,0.]],dtype=np.float32)
    a=sparse.csr_matrix([[0.,0.]],dtype=np.float32)
    x=combine(n,a)
    assert (x@x.T).toarray()[0,0]==pytest.approx(.5)
    with pytest.raises(ValueError):combine(n,a,1)


def test_witness_pruning_preserves_edges_and_forced_context():
    rng=np.random.default_rng(9)
    full=sparse.random(60,20,density=.6,random_state=rng,dtype=np.float32).tocsr()
    targets=sparse.random(25,20,density=.5,random_state=rng,dtype=np.float32).tocsr()
    emit=[1,5,19]
    kw=dict(k=3,floor=.1,emit_indices=emit,emit_index_transposed=full[emit].T.tocsr(),force_rows=[0,9])
    expected=retrieve_batch(targets,full.T.tocsr(),**kw)
    got=retrieve_batch(targets,full.T.tocsr(),witness_index_transposed=full.T.tocsr(),**kw)
    assert expected.hits==got.hits
    assert got.targets_scored_globally < expected.targets_scored_globally
    for i in [0,9]:
        assert next(x for x in got.contexts if x['target_row']==i)==next(x for x in expected.contexts if x['target_row']==i)


def test_witness_preserves_equal_score_boundary_ties():
    full=sparse.csr_matrix(np.ones((20,1),dtype=np.float32))
    targets=sparse.csr_matrix([[1.]],dtype=np.float32)
    got=retrieve_batch(targets,full.T.tocsr(),k=1,floor=.1,emit_indices=[19],
        emit_index_transposed=full[[19]].T.tocsr(),witness_index_transposed=full[:10].T.tocsr())
    assert len(got.hits)==1 and got.hits[0]['owner_row']==19


def test_context_boundary_ties_below_emission_floor_are_complete():
    full=sparse.csr_matrix(np.full((20,1),.1,dtype=np.float32))
    got=retrieve_batch(sparse.csr_matrix([[1.]],dtype=np.float32),full.T.tocsr(),k=1,floor=.3)
    assert not got.hits and len(got.contexts[0]['top_owners'])==20
