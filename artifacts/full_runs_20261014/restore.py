"""Restore archived experiment data after validating its checksums."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
import tempfile


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024):digest.update(data)
    return digest.hexdigest()


def extract(tar,destination):
    root=destination.resolve()
    members=tar.getmembers()
    for member in members:
        if not (member.isfile() or member.isdir() or member.islnk()):
            raise ValueError('Unsupported archive entry')
        (root/member.name).resolve().relative_to(root)
        if member.name.startswith('/') or not member.name.startswith(('runs/','models/ckpts/','experiments/gpt_smoke/')):
            raise ValueError('Unexpected archive path')
        if member.islnk():
            (root/member.linkname).resolve().relative_to(root)
            if member.linkname.startswith('/'):raise ValueError('Absolute link')
        member.mode &= 0o777
    options={'filter':'data'} if 'filter' in inspect.signature(tar.extractall).parameters else {}
    tar.extractall(root,members=members,**options)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args();args.dest.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    m=json.loads((here/'manifest.json').read_text())
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as f:
        for part in m['parts']:
            path=here/part['path']
            assert path.stat().st_size==part['size'] and sha(path)==part['sha256']
            with path.open('rb') as source:
                while data:=source.read(1024*1024):f.write(data)
        f.flush();archive=Path(f.name)
        assert sha(archive)==m['archive_sha256']
        with tarfile.open(archive,'r:gz') as tar:extract(tar,args.dest)
    manifests=[m]
    for name,manifest in [('legacy_model_checkpoints.tar.gz','legacy_checkpoints_manifest.json'),
                          ('legacy_smoke.tar.gz','legacy_smoke_manifest.json')]:
        details=json.loads((here/manifest).read_text());assert sha(here/name)==details['archive_sha256']
        with tarfile.open(here/name,'r:gz') as tar:extract(tar,args.dest)
        manifests.append(details)
    count=0
    for details in manifests:
        for entry in details['files']:
            path=args.dest/entry['path'];assert path.stat().st_size==entry['size'] and sha(path)==entry['sha256'];count+=1
    print(f'Restored and verified {count} files in {args.dest}')


if __name__=='__main__':main()
