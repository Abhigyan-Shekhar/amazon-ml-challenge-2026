"""Create a fresh, disjoint uniform confirmation sample; no model selection here."""
import csv,json,random
from pathlib import Path
old=Path('artifacts/validation/full_audit/sample_s1.jsonl')
exclude={json.loads(line)['entity_id'] for line in old.read_text().splitlines()};rng=random.Random(2028);sample=[];seen=0
root=Path('/Users/abhigyanshekhar/Downloads/train')
with (root/'train_source1.tsv').open(newline='') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  if row['entity_id'] in exclude:continue
  seen+=1
  if len(sample)<2000:sample.append(row)
  else:
   j=rng.randrange(seen)
   if j<2000:sample[j]=row
byid={r['entity_id']:r for r in sample}
with (root/'train_ground_truth.tsv').open(newline='') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  if row['source1_entity_id'] in byid:byid[row['source1_entity_id']]['matches']=row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
assert not exclude&set(byid)
out=Path('artifacts/validation/confirmation');out.mkdir(parents=True,exist_ok=False)
with (out/'sample_s1.jsonl').open('w') as f:
 for r in sample:
  assert 'matches' in r
  f.write(json.dumps(r,ensure_ascii=False)+'\n')
(out/'manifest.json').write_text(json.dumps({'seed':2028,'sample_size':2000,'excluded_prior_s1':len(exclude),'purpose':'untouched confirmation; no threshold/model fitting'},indent=2))
print('Created',len(sample),'disjoint confirmation S1s',flush=True)
