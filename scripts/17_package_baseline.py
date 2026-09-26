"""Official full-ID validation and packaging. Never submits to a leaderboard."""
import argparse,contextlib,hashlib,importlib.metadata,io,json,platform,shutil,sys,time,zipfile
from pathlib import Path
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.validate_submission import validate

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--test-dir',type=Path,required=True);a=p.parse_args()
for name in ['matching_results.tsv','candidate_pairs.tsv','result.json','final_manifest.json']:
    if not (a.run/name).is_file():raise ValueError('Incomplete run: '+name)
start=time.monotonic();capture=io.StringIO()
with contextlib.redirect_stdout(capture):errors,warnings=validate(str(a.run/'matching_results.tsv'),str(a.run/'candidate_pairs.tsv'),str(a.test_dir),check_ids=True)
report={'errors':errors,'warnings':warnings,'check_ids':True,'validator_sha256':hashlib.sha256(Path('utils/validate_submission.py').read_bytes()).hexdigest(),'stdout':capture.getvalue(),'runtime_seconds':time.monotonic()-start}
(a.run/'official_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if errors or warnings:raise ValueError('Refusing packaging: official errors/warnings must be resolved')
manifest=json.loads((a.run/'final_manifest.json').read_text());bundle=joblib.load(manifest['checkpoint_path'])
if manifest.get('structured_model'):
    shutil.copy(manifest['checkpoint_path'],a.run/'model.joblib')
    parameter_file='model.joblib'
else:
    model=bundle['cheap_model'];scaler=model.named_steps['standardscaler'];lr=model.named_steps['logisticregression']
    export={'features':manifest['feature_config'],'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'threshold':manifest['threshold']}
    (a.run/'classifier_parameters.json').write_text(json.dumps(export,indent=2))
    parameter_file='classifier_parameters.json'
# Add packaging provenance separately; the pre-inference freeze remains unchanged.
versions=['numpy','pandas','scikit-learn','scipy','joblib']
if manifest.get('model_name','').startswith('xgboost.'):
    versions.append('xgboost')
package={'frozen_manifest':manifest,'source_state_note':'Source hashes identify the code included in this package. The archived v1 checkpoint remains separate.','python':platform.python_version(),'package_versions':{p:importlib.metadata.version(p) for p in versions},'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in ['src','scripts','utils'] for p in sorted(Path(base).rglob('*.py'))},'input_manifest':json.loads(Path('artifacts/reports/input_manifest.json').read_text()),'official_validation':report,'submission_recommendation':('Validated local package; explicit human approval required before any leaderboard upload. France remains unvalidated.' if manifest.get('uses_fuzzy') else 'Hold: stronger validated CPU checkpoint available. No upload approved or performed.')}
if manifest.get('uses_fuzzy'):
    checkpoint_report=Path(manifest['checkpoint_path']).with_name('validation.json')
    measured=json.loads(checkpoint_report.read_text()) if checkpoint_report.exists() else {}
    confirmation_path=Path('artifacts/validation/confirmation/missing_address_v2.json') if manifest.get('neutral_missing_address') else Path('artifacts/reports/fuzzy_confirmation.json')
    package['independent_confirmation']=json.loads(confirmation_path.read_text()) if confirmation_path.exists() else measured.get('confirmation')
    package['package_versions']['rapidfuzz']=importlib.metadata.version('rapidfuzz')
package['output_sha256']={}
for name in ['matching_results.tsv','candidate_pairs.tsv']:
 with (a.run/name).open('rb') as f:package['output_sha256'][name]=hashlib.file_digest(f,'sha256').hexdigest()
(a.run/'package_manifest.json').write_text(json.dumps(package,indent=2))
archive=a.run.parent/(a.run.name+'_submission.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for name in ['matching_results.tsv','candidate_pairs.tsv']:z.write(a.run/name,arcname='output/'+name)
 for name in ['final_manifest.json','package_manifest.json','official_validation.json',parameter_file]:z.write(a.run/name,arcname=name)
 z.write('Documentation_template.md',arcname='Documentation_template.md')
 z.write('artifacts/methodology.md',arcname='methodology.md')
 for base in ['src','scripts','utils']:
  for source in sorted(Path(base).rglob('*.py')):z.write(source,arcname='code/business_entity_resolution/'+str(source))
 for name in ['README.md','requirements.txt']:z.write(name,arcname=name)
 z.write('config.example.json',arcname='code/business_entity_resolution/config.example.json')
print('Created',archive,'bytes',archive.stat().st_size,flush=True)
