"""Sample the cost of global char retrieval; never launch full-corpus matrices."""
import csv,json,time,sys
from itertools import islice
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize
start=time.monotonic()
with open('/Users/abhigyanshekhar/Downloads/train/train_source2.tsv') as f:
    records=list(islice(csv.DictReader(f,delimiter='\t'),20000))
texts=[normalize(r['business_name']+' '+r['business_address']) for r in records]
v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),dtype=np.float32,max_features=300000)
matrix=v.fit_transform(texts)
fit_seconds=time.monotonic()-start
start=time.monotonic();v.transform(texts);transform_seconds=time.monotonic()-start
start=time.monotonic();product=matrix[:100]@matrix.T;query_seconds=time.monotonic()-start
full_targets=5034616+5285603;full_queries=2206821
report={'sample_rows':len(texts),'sample_nnz':matrix.nnz,'average_nnz':matrix.nnz/len(texts),'sample_fit_seconds':fit_seconds,'sample_transform_seconds':transform_seconds,'sample_query_100_seconds':query_seconds,'projected_target_sparse_matrix_gib_lower_bound':(matrix.nnz/len(texts)*8+4)*full_targets/2**30,'projected_transform_seconds_linear':transform_seconds*full_targets/len(texts),'projected_full_query_seconds_linear':query_seconds*full_targets/len(texts)*full_queries/100,'caveat':'rough linear projections, exclude strings, vocabulary, temporary products and candidate storage; unseen ngrams change sparsity','decision':'use selective lexical blocking before within-block char scoring on this 16 GiB CPU host'}
Path('artifacts/validation/char_scale_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
