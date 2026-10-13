"""Benchmark complete self-play PPO iterations in isolated CPU worker processes."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
from environment import STADIUM
from model import Policy
from train import (create_training_env,T,EPOCHS,N_MB,CLIP,VF_COEF,MAX_GRAD,LAM,LR,HALF_LIFE_S,ENT0)


def worker(envs, threads, iterations):
    torch.set_num_threads(threads)
    torch.manual_seed(42);np.random.seed(42)
    env=create_training_env(envs,1,1,STADIUM,8,seed=42)
    model=Policy(env.obs_dim,env.action_space);opp=Policy(env.obs_dim,env.action_space)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    obs=env.reset()
    so=torch.zeros(T,envs,env.obs_dim)
    sa=torch.zeros(T,envs,3,dtype=torch.long)
    sl=torch.zeros(T,envs);sv=torch.zeros(T,envs);sr=torch.zeros(T,envs);sd=torch.zeros(T,envs)
    gamma=math.exp(math.log(.5)/(7.5*HALF_LIFE_S))
    rows=[]
    for iteration in range(iterations+2):
        start=time.perf_counter();opp.load_state_dict(model.state_dict())
        for t in range(T):
            o=torch.as_tensor(obs,dtype=torch.float32)
            with torch.no_grad():
                action,logp,value=model.act_train(o[:,0])
                other=opp.act(o[:,1])
            so[t],sa[t],sl[t],sv[t]=o[:,0],action,logp,value
            full=np.stack([action.numpy(),other.numpy()],axis=1)
            obs,rew,term,trunc=env.step(full)
            sr[t]=torch.as_tensor(rew[:,0]);sd[t]=torch.as_tensor((term|trunc).astype(np.float32))
        rollout_end=time.perf_counter()
        with torch.no_grad():last=model._logits(torch.as_tensor(obs[:,0],dtype=torch.float32))[1]
        adv=torch.zeros_like(sr);gae=torch.zeros(envs)
        for t in reversed(range(T)):
            mask=1-sd[t];nv=last if t==T-1 else sv[t+1]
            delta=sr[t]+gamma*nv*mask-sv[t]
            gae=delta+gamma*LAM*mask*gae;adv[t]=gae
        ret=(adv+sv).flatten();a=adv.flatten();a=(a-a.mean())/(a.std()+1e-8)
        bo,ba,bl=so.reshape(-1,env.obs_dim),sa.reshape(-1,3),sl.flatten()
        idx=np.arange(T*envs);mb=len(idx)//N_MB
        gae_end=time.perf_counter()
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for offset in range(0,len(idx),mb):
                mi=torch.as_tensor(idx[offset:offset+mb])
                logp,ent,val=model.evaluate(bo[mi],ba[mi])
                ratio=(logp-bl[mi]).exp()
                loss=-torch.minimum(ratio*a[mi],ratio.clamp(1-CLIP,1+CLIP)*a[mi]).mean()
                loss+=VF_COEF*.5*(val-ret[mi]).square().mean()-ENT0*ent.mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD);opt.step()
        end=time.perf_counter()
        if iteration>=2:rows.append(dict(rollout_s=rollout_end-start,gae_s=gae_end-rollout_end,update_s=end-gae_end,total_s=end-start))
    total=sum(r['total_s'] for r in rows)
    return dict(envs=envs,torch_threads=threads,rayon_threads=os.environ.get('RAYON_NUM_THREADS','default'),
      measured_iterations=iterations,warmup_iterations=2,samples_per_iteration=T*envs,
      samples_per_second=T*envs*iterations/total,median_iteration_s=float(np.median([r['total_s'] for r in rows])),
      mean_rollout_s=float(np.mean([r['rollout_s'] for r in rows])),mean_update_s=float(np.mean([r['update_s'] for r in rows])),
      max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,iterations=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--envs',type=int,default=512)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--iterations',type=int,default=6)
    ap.add_argument('--out',type=Path,default=Path('experiments/gpt_cpu_benchmark'))
    args=ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.envs,args.threads,args.iterations)));return
    args.out.mkdir(parents=True,exist_ok=True)
    report={'cpu':'AMD Ryzen 7 5700U, 8 cores / 16 logical CPUs','method':'Full PPO iteration, 64 steps, 6 epochs, 4 minibatches, neural self-play, corrected map. Two warmups; checkpoints/evaluation excluded. Desktop applications remained running.','runs':[]}
    # Reverse env order in the second pass reduces order/thermal bias.
    settings=[(128,2),(256,2),(512,2),(512,4),(512,8),(512,2),(256,2),(128,2)]
    for n,threads in settings:
        command=[sys.executable,__file__,'--worker','--envs',str(n),'--threads',str(threads),'--iterations',str(args.iterations)]
        output=subprocess.run(command,check=True,capture_output=True,text=True)
        result=json.loads(output.stdout);report['runs'].append(result)
        (args.out/'results.json').write_text(json.dumps(report,indent=2))
        print(f'{n} envs / {threads} torch threads: {result["samples_per_second"]:.0f} samples/s, {result["median_iteration_s"]:.3f} s/iteration, {result["max_rss_mib"]:.0f} MiB',flush=True)

if __name__=='__main__':main()
