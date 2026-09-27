"""Official full-ID validation and packaging. Never submits to a leaderboard."""
import argparse,contextlib,hashlib,importlib.metadata,io,json,platform,shutil,sys,time,zipfile
from pathlib import Path
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.validate_submission import validate

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--test-dir',type=Path,required=True)
p.add_argument('--team-name',default='BlackList',help='Team name for official submission zip (default: BlackList)')
p.add_argument('--archive-name',default=None,help='Explicit zip archive filename')
a=p.parse_args()

for name in ['matching_results.tsv','candidate_pairs.tsv','result.json','final_manifest.json']:
    if not (a.run/name).is_file():raise ValueError('Incomplete run: '+name)
start=time.monotonic();capture=io.StringIO()
with contextlib.redirect_stdout(capture):errors,warnings=validate(str(a.run/'matching_results.tsv'),str(a.run/'candidate_pairs.tsv'),str(a.test_dir),check_ids=True)
report={'errors':errors,'warnings':warnings,'check_ids':True,'validator_sha256':hashlib.sha256(Path('utils/validate_submission.py').read_bytes()).hexdigest(),'stdout':capture.getvalue(),'runtime_seconds':time.monotonic()-start}
(a.run/'official_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if errors or warnings:raise ValueError('Refusing packaging: official errors/warnings must be resolved')
manifest=json.loads((a.run/'final_manifest.json').read_text());bundle=joblib.load(manifest['checkpoint_path'])
if manifest['model_name']=='structured_cheap_tree':
    shutil.copy(manifest['checkpoint_path'],a.run/'model.joblib')
    parameter_file='model.joblib'
else:
    model=bundle['cheap_model'];scaler=model.named_steps['standardscaler'];lr=model.named_steps['logisticregression']
    export={'features':manifest['feature_config'],'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'threshold':manifest['threshold']}
    (a.run/'classifier_parameters.json').write_text(json.dumps(export,indent=2))
    parameter_file='classifier_parameters.json'
# Add packaging provenance separately; the pre-inference freeze remains unchanged.
package={'frozen_manifest':manifest,'source_state_note':'Source hashes identify the code included in this package. The archived v1 checkpoint remains separate.','python':platform.python_version(),'package_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib']},'source_sha256':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for base in ['src','scripts','utils'] for p in sorted(Path(base).rglob('*.py'))},'input_manifest':json.loads(Path('artifacts/reports/input_manifest.json').read_text()),'official_validation':report,'submission_recommendation':('Validated local package; explicit human approval required before any leaderboard upload. France remains unvalidated.' if manifest.get('uses_fuzzy') else 'Hold: stronger validated CPU checkpoint available. No upload approved or performed.')}
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

zip_name = a.archive_name if a.archive_name else f"{a.team_name}_submission.zip"
archive = a.run.parent / zip_name

# Self-contained reproduction README for code/business_entity_resolution/
code_readme = (
    "# Business Entity Resolution Pipeline — Amazon ML Challenge 2026\n\n"
    f"**Team Name:** {a.team_name}\n"
    "**Approach:** Structured candidate blocking + HistGradientBoosting with missing-address handling\n"
    "**Selected Threshold:** 0.585\n\n"
    "## 1. Environment Setup\n\n"
    "Requires Python 3.10+ (tested on Python 3.10 / 3.11 / 3.13):\n"
    "```bash\n"
    "pip install -r requirements.txt\n"
    "```\n\n"
    "## 2. End-to-End Reproduction (Generating Outputs)\n\n"
    "To regenerate `matching_results.tsv` and `candidate_pairs.tsv` from test data:\n\n"
    "```bash\n"
    "PYTHONHASHSEED=2026 python scripts/18_full_structured.py \\\n"
    "    --test-dir /path/to/dataset/test \\\n"
    "    --model-dir . \\\n"
    "    --tree-dir . \\\n"
    "    --max-key-frequency 10 \\\n"
    "    --output output \\\n"
    "    --budget-seconds 4200 \\\n"
    "    --memory-gib 10\n"
    "```\n\n"
    "## 3. Official Output Validation\n\n"
    "To validate output format, column names, singleton handling, and entity IDs:\n\n"
    "```bash\n"
    "python utils/validate_submission.py \\\n"
    "    --matching output/matching_results.tsv \\\n"
    "    --candidate output/candidate_pairs.tsv \\\n"
    "    --test-dir /path/to/dataset/test \\\n"
    "    --check-ids\n"
    "```\n"
)

code_base = 'code/business_entity_resolution'

with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    # 1. output/ folder with both required TSVs
    for name in ['matching_results.tsv', 'candidate_pairs.tsv']:
        z.write(a.run / name, arcname=f"output/{name}")

    # 2. Documentation_template.md at zip root
    z.write('Documentation_template.md', arcname='Documentation_template.md')

    # 3. code/business_entity_resolution/ containing self-contained runnable pipeline
    z.writestr(f"{code_base}/README.md", code_readme)
    if Path('requirements.txt').exists():
        z.write('requirements.txt', arcname=f"{code_base}/requirements.txt")
    if Path('config.example.json').exists():
        z.write('config.example.json', arcname=f"{code_base}/config.example.json")

    # Write all Python source code (strictly using forward slashes)
    for base in ['src', 'scripts', 'utils']:
        for source in sorted(Path(base).rglob('*.py')):
            z.write(source, arcname=f"{code_base}/{source.as_posix()}")

    # Include model parameter file and manifests inside code/business_entity_resolution/
    for name in ['final_manifest.json', 'package_manifest.json', 'official_validation.json', parameter_file]:
        if (a.run / name).exists():
            z.write(a.run / name, arcname=f"{code_base}/{name}")

print('Created', archive, 'bytes', archive.stat().st_size, flush=True)
