"""Build a full-S1 sparse index, stream targets, and export reverse candidate edges.

No labels are loaded. --sample restricts emitted owners, never competitors.
Index/retrieval artifacts have completion markers and immutable input hashes.
"""
import argparse
import csv
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import psutil
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blocking.reverse import VERSION, vectorizers, combine, transform, retrieve_batch
from src.normalize import normalize, VERSION as NORMALIZATION_VERSION


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def records(path):
    with Path(path).open(newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def snapshot(start):
    return {'seconds': time.monotonic()-start, 'rss_gib': psutil.Process().memory_info().rss/2**30}


def guard(start, memory, seconds):
    state = snapshot(start)
    if state['rss_gib'] > memory or state['seconds'] > seconds:
        raise RuntimeError('Runtime/memory gate exceeded; partial output is not complete')
    return state


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    ids = [r['entity_id'] for r in records(args.s1)]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate S1 IDs')
    np.save(args.output/'ids.npy', np.array(ids))
    nv, av = vectorizers()
    print(json.dumps({'stage': 'name_index', 's1': len(ids)}), flush=True)
    name = nv.fit_transform(normalize(r['business_name']) for r in records(args.s1))
    print(json.dumps({'stage': 'address_index', **guard(start,args.memory_gib,args.budget_seconds)}), flush=True)
    address = av.fit_transform(normalize(r['business_address']) for r in records(args.s1))
    matrix = combine(name, address, args.name_weight)
    del name, address
    gc.collect()
    guard(start,args.memory_gib,args.budget_seconds)
    # One CSR transpose is enough for both global retrieval and emission-subset gating.
    transposed = matrix.T.tocsr()
    del matrix
    gc.collect()
    sparse.save_npz(args.output/'index_transposed.npz', transposed, compressed=False)
    joblib.dump({'name': nv, 'address': av}, args.output/'vectorizers.joblib')
    manifest = {'version': VERSION, 'normalization': NORMALIZATION_VERSION,
                'name_weight': args.name_weight, 's1_count': len(ids), 's1_sha256': sha(args.s1),
                'idf_fit': 'All S1 rows, without labels; no hard country partition',
                'name_vocabulary': len(nv.vocabulary_), 'address_vocabulary': len(av.vocabulary_),
                'nnz': int(transposed.nnz), 'shape': list(transposed.shape),
                'artifacts': {p: sha(args.output/p) for p in ('ids.npy','index_transposed.npz','vectorizers.joblib')},
                'versions': {p:version(p) for p in ('numpy','scipy','scikit-learn','sparse-dot-topn')},
                **guard(start,args.memory_gib,args.budget_seconds), 'complete':True}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


def retrieve(args):
    start = time.monotonic()
    manifest = json.loads((args.index/'manifest.json').read_text())
    if (not manifest.get('complete') or manifest['version'] != VERSION or
            manifest.get('normalization') != NORMALIZATION_VERSION):
        raise ValueError('Incomplete or incompatible index')
    for filename, expected in manifest['artifacts'].items():
        if sha(args.index/filename) != expected:
            raise ValueError('Index artifact hash mismatch: '+filename)
    ids = np.load(args.index/'ids.npy',allow_pickle=False)
    index = sparse.load_npz(args.index/'index_transposed.npz')
    vec = joblib.load(args.index/'vectorizers.joblib')  # locally generated, hash-checked artifact
    emit, gate = None, None
    if args.sample:
        sample = [json.loads(l) for l in args.sample.read_text().splitlines() if l.strip()]
        wanted = {r['entity_id'] for r in sample}
        if not wanted or len(wanted) != len(sample):
            raise ValueError('Empty or duplicate sample S1')
        emit = np.flatnonzero(np.isin(ids, list(wanted)))
        if len(emit) != len(wanted):
            raise ValueError('Unknown sampled S1')
        gate = index[:,emit].tocsr()
    witness = index[:, ::args.witness_stride].tocsr() if emit is not None and args.witness_stride else None
    forward_targets=set()
    if args.forward_pairs:
        with args.forward_pairs.open(newline='') as f:
            forward_targets={r['target_entity_id'] for r in csv.DictReader(f,delimiter='\t')}
    identity = {'index': sha(args.index/'manifest.json'),
                'sample': sha(args.sample) if args.sample else None,
                'forward': sha(args.forward_pairs) if args.forward_pairs else None,
                's2': sha(args.s2), 's3': sha(args.s3),
                'k': args.k, 'floor': args.floor, 'witness_stride': args.witness_stride,
                'version': VERSION, 'max_targets_per_source': args.max_targets_per_source,
                'code': {p.name: sha(p) for p in (Path(__file__), Path(__file__).resolve().parents[1]/'src/blocking/reverse.py')}}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoint = args.output/'checkpoint.json'
    saved = None
    if args.resume:
        if (args.output/'manifest.json').exists():
            raise ValueError('Output already finalized; choose a new directory')
        saved = json.loads(checkpoint.read_text())
        if saved['identity'] != identity:
            raise ValueError('Resume input/configuration mismatch')
    scanned = qualified = emitted = 0
    peak = 0
    timings = {'transform':0., 'retrieve':0., 'write':0.}
    partial = args.output/'reverse_pairs.jsonl.partial'
    context_path=args.output/'target_context.jsonl.partial'
    done = {'s2': 0, 's3': 0}
    if saved:
        done = saved['done']
        scanned, qualified, emitted = (saved[x] for x in ('scanned', 'qualified', 'emitted'))
        timings = saved['timings']
        peak = saved['peak_observed_rss_gib']
        # Include elapsed active work across resumptions, excluding downtime.
        start -= saved['seconds']
        for path, offset in ((partial, saved['pairs_offset']), (context_path, saved['context_offset'])):
            if not path.exists() and path.with_suffix('').exists():
                path.with_suffix('').rename(path)  # Interrupted finalization before manifest.
            if path.stat().st_size < offset:
                raise ValueError('Partial file shorter than committed checkpoint')
            with path.open('r+b') as stream:
                stream.truncate(offset)
    with partial.open('a' if saved else 'x') as output, context_path.open('a' if saved else 'x') as context_output:
        if not saved:
            checkpoint.with_suffix('.tmp').write_text(json.dumps({'identity': identity,
                'done': done, 'scanned': 0, 'qualified': 0, 'emitted': 0, 'timings': timings,
                'peak_observed_rss_gib': 0, 'seconds': snapshot(start)['seconds'],
                'pairs_offset': 0, 'context_offset': 0}))
            checkpoint.with_suffix('.tmp').replace(checkpoint)
        for source,path in [('s2',args.s2),('s3',args.s3)]:
            batch=[]; count=done[source]
            def flush(batch):
                nonlocal scanned,qualified,emitted,peak
                t=time.monotonic()
                matrix=transform(batch,vec['name'],vec['address'],manifest['name_weight'])
                timings['transform']+=time.monotonic()-t
                t=time.monotonic()
                result=retrieve_batch(matrix,index,args.k,args.floor,args.threads,emit,gate,
                    [i for i,r in enumerate(batch) if r['entity_id'] in forward_targets], witness)
                timings['retrieve']+=time.monotonic()-t
                t=time.monotonic()
                # Export context only for targets needed by forward or emitted reverse edges.
                needed={h['target_row'] for h in result.hits}
                for context in result.contexts:
                    ti=context.pop('target_row')
                    if ti not in needed and batch[ti]['entity_id'] not in forward_targets:continue
                    owner=context.pop('best_owner_row')
                    context['best_s1_id']=str(ids[owner]) if owner is not None else None
                    context['target_entity_id']=batch[ti]['entity_id']
                    for owner in context['top_owners']:
                        owner['source1_entity_id']=str(ids[owner.pop('owner_row')])
                    context_output.write(json.dumps(context)+'\n')
                for h in result.hits:
                    row=batch[h.pop('target_row')]; owner=str(ids[h.pop('owner_row')])
                    payload={'source1_entity_id':owner,'target_entity_id':row['entity_id'],
                             'target_name':row['business_name'],'target_address':row['business_address'],
                             'target_country':row['country'],'target_source':source,
                             'retrieval_routes':['reverse_sparse'],**h}
                    output.write(json.dumps(payload,ensure_ascii=False)+'\n')
                emitted+=len(result.hits);qualified+=result.targets_scored_globally;scanned+=len(batch)
                timings['write']+=time.monotonic()-t
                state=guard(start,args.memory_gib,args.budget_seconds);peak=max(peak,state['rss_gib'])
                progress={'scanned':scanned,'globally_scored':qualified,'emitted':emitted,**state}
                done[source] += len(batch)
                output.flush(); context_output.flush()
                os.fsync(output.fileno()); os.fsync(context_output.fileno())
                state_record = {'identity': identity, 'done': done, 'scanned': scanned,
                    'qualified': qualified, 'emitted': emitted, 'timings': timings,
                    'peak_observed_rss_gib': peak, 'seconds': state['seconds'],
                    'pairs_offset': output.tell(), 'context_offset': context_output.tell()}
                checkpoint.with_suffix('.tmp').write_text(json.dumps(state_record))
                checkpoint.with_suffix('.tmp').replace(checkpoint)
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                if scanned % (args.batch_size*10)==0: print(json.dumps(progress),flush=True)
            for row_number, row in enumerate(records(path)):
                if row_number < done[source]:
                    continue
                if args.max_targets_per_source and count>=args.max_targets_per_source:break
                batch.append(row);count+=1
                if len(batch)==args.batch_size:flush(batch);batch=[]
            if batch:flush(batch)
    partial.rename(args.output/'reverse_pairs.jsonl')
    context_path.rename(args.output/'target_context.jsonl')
    report={'version':VERSION,'complete':not bool(args.max_targets_per_source),
            'benchmark_only':bool(args.max_targets_per_source),'s1_competitors':len(ids),
            'emitted_s1_count':len(emit) if emit is not None else len(ids),
            'sample_sha256':sha(args.sample) if args.sample else None,
            'index_manifest_sha256':sha(args.index/'manifest.json'),
            'target_hashes':{p.name:sha(p) for p in (args.s2,args.s3)},
            'k':args.k,'score_floor_strict':args.floor,'threads':args.threads,'witness_stride':args.witness_stride,
            'tie_policy':'competition rank; retain all top-k boundary ties',
            'scanned_targets':scanned,'globally_scored_targets':qualified,'edges':emitted,
            'timings':timings,'peak_observed_rss_gib':peak,'identity':identity,'scanned_by_source':done,**snapshot(start),
            'forward_pairs_sha256':sha(args.forward_pairs) if args.forward_pairs else None,
            'context_sha256':sha(args.output/'target_context.jsonl'),
            'pairs_sha256':sha(args.output/'reverse_pairs.jsonl')}
    (args.output/'manifest.json.tmp').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'manifest.json.tmp').replace(args.output/'manifest.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--s1',type=Path,required=True);b.add_argument('--name-weight',type=float,default=.5)
    r=sub.add_parser('retrieve');r.add_argument('--index',type=Path,required=True)
    for name in ('s2','s3'):r.add_argument('--'+name,type=Path,required=True)
    r.add_argument('--forward-pairs',type=Path,help='Finalized forward TSV: also export all-S1 context for these targets')
    r.add_argument('--sample',type=Path);r.add_argument('--k',type=int,default=8);r.add_argument('--floor',type=float,default=.35)
    r.add_argument('--threads',type=int,default=2);r.add_argument('--batch-size',type=int,default=1024)
    r.add_argument('--resume',action='store_true',help='Resume verified partial output at last committed batch')
    r.add_argument('--witness-stride',type=int,default=32,help='Safe competitor prescreen; 0 disables')
    r.add_argument('--max-targets-per-source',type=int,default=0)
    for parser in (b,r):
        parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memory-gib',type=float,default=5)
        parser.add_argument('--budget-seconds',type=float,default=3600)
    a=p.parse_args()
    if a.memory_gib <= 0 or a.budget_seconds <= 0:
        p.error('memory and time budgets must be positive')
    if a.command == 'retrieve' and (a.batch_size < 1 or a.max_targets_per_source < 0 or
            a.k < 1 or a.threads < 1 or a.witness_stride < 0 or not 0 <= a.floor < 1):
        p.error('invalid retrieval budget, batch size, threads, stride, or floor')
    (build if a.command=='build' else retrieve)(a)
