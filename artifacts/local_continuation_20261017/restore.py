"""Restore a verified delta of public experiment data; never starts a trainer."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import tarfile
import tempfile
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024):h.update(data)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dest',required=True,type=Path);args=ap.parse_args()
    root=args.dest.resolve();root.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent;m=json.loads((here/'manifest.json').read_text())
    with tempfile.NamedTemporaryFile() as temp:
        for part in m['parts']:
            p=here/part['path']
            if sha(p)!=part['sha256'] or p.stat().st_size!=part['size']:raise ValueError('Bad part checksum')
            with p.open('rb') as f:shutil.copyfileobj(f,temp)
        temp.flush()
        if sha(Path(temp.name))!=m['archive_sha256']:raise ValueError('Bad archive checksum')
        with tarfile.open(temp.name,'r:gz') as tf:
            members=tf.getmembers()
            for member in members:
                if not member.isfile() or not member.name.startswith('runs/') or Path(member.name).is_absolute():raise ValueError('Unsafe entry')
                (root/member.name).resolve().relative_to(root)
                member.mode &= 0o777
            opts={'filter':'data'} if 'filter' in inspect.signature(tf.extractall).parameters else {}
            tf.extractall(root,members=members,**opts)
    for item in m['files']:
        p=root/item['path']
        if p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:raise ValueError('Restored file mismatch')
    print('Verified and restored',len(m['files']),'files; no process was started')

if __name__=='__main__':main()
