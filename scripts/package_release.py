"""Build a source-only release directory with an explicit file allowlist."""
import argparse
import hashlib
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]; target=Path(args.output).resolve()
    if target==root or root in target.parents:
        raise ValueError('Release staging must be outside the project directory')
    if target.exists(): raise ValueError('Use a new release staging directory')
    target.mkdir(parents=True)
    explicit=['README.md','VALIDATION.md','LICENSE','THIRD_PARTY.md','pyproject.toml','.gitignore','requirements-tested.txt']
    files=[root/name for name in explicit]
    for folder in ('src','tests','configs','ontology','adapters','docs','scripts','.github'):
        files += [path for path in (root/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts
                  and not any(part.endswith('.egg-info') for part in path.parts)
                  and path.suffix not in ('.pyc','.pt','.pth','.joblib')]
    for name in ('unit_test_log.txt','utterance_verification.json','vendor_provenance.json','installed_package_verification.json'):
        path=root/'examples'/name
        if path.is_file(): files.append(path)
    hashes=[]
    for path in sorted(set(files)):
        rel=path.relative_to(root); dest=target/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        hashes.append(hashlib.sha256(dest.read_bytes()).hexdigest()+'  '+str(rel))
    (target/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')
    print('Source release:',target,'files:',len(hashes))

if __name__=='__main__': main()
