"""Publish a coherent local bundle from one durable resume checkpoint (no Git mutation)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from training_checkpoint import atomic_torch_save
from telemetry import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dest',type=Path,default=Path('models/sprint1_latest'))
    args=parser.parse_args()
    # Atomic replace by trainer means this open reads a complete old or new generation.
    checkpoint=torch.load(args.run/'resume.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('schema_version')!=1:raise ValueError('Unknown checkpoint schema')
    destination=args.dest;destination.mkdir(parents=True,exist_ok=True)
    meta=dict(checkpoint['config'],stadium=str(destination/'stadium.hbs'),
              it=checkpoint['best_iteration'],evaluation_score=checkpoint['best'])
    atomic_torch_save(checkpoint['best_model'],destination/'model.pt')
    atomic_json(destination/'meta.json',meta,durable=True)
    atomic_torch_save(checkpoint['model'],destination/'latest.pt')
    atomic_json(destination/'meta_latest.json',dict(meta,it=checkpoint['iteration'],evaluation_score=None),durable=True)
    atomic_torch_save(checkpoint,destination/'resume.pt')
    (destination/'stadium.hbs').write_text(checkpoint['stadium_text'])
    provenance=dict(iteration=checkpoint['iteration'],best_iteration=checkpoint['best_iteration'],
                    best_score=checkpoint['best'],source=str(args.run),
                    resume_sha256=hashlib.sha256((destination/'resume.pt').read_bytes()).hexdigest(),
                    resume_semantics=checkpoint['config']['resume_semantics'])
    atomic_json(destination/'backup.json',provenance,durable=True)
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
