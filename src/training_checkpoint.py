"""Durable iteration-boundary checkpoints. Native matches resume with fresh resets."""
import os
from pathlib import Path
import torch
import numpy as np
from telemetry import atomic_replace


def atomic_torch_save(value, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        torch.save(value,handle)
        handle.flush();os.fsync(handle.fileno())
    atomic_replace(temporary,path)
    if hasattr(os,'O_DIRECTORY'):
        directory=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)


def random_state(rng):
    global_state=np.random.get_state()
    return dict(torch=torch.get_rng_state(),numpy=[global_state[0],global_state[1].tolist(),
                global_state[2],global_state[3],global_state[4]],
                generator=rng.bit_generator.state,
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_state(state,rng):
    torch.set_rng_state(state['torch'].cpu())
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    rng.bit_generator.state=state['generator']
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
