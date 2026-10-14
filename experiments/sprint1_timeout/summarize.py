"""Summarize completed arms; paired seeds are the replication unit, not episodes."""
import json
from pathlib import Path
import statistics
import sys
import torch


def summarize(root):
    manifest = json.loads((root/'manifest.json').read_text())
    results = []
    checks = []
    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    for name in manifest['completed']:
        folder = root/name
        config = json.loads((folder/'config.json').read_text())
        rows = read_jsonl(folder/'training.jsonl')
        evals = read_jsonl(folder/'evaluations.jsonl')
        best = json.loads((folder/'meta.json').read_text())['it']
        assert len(rows) == manifest['iters'], name
        assert rows[-1]['samples'] == manifest['samples_per_arm_seed'], name
        assert config['warm_start_sha256'] == manifest['checkpoint_sha256'], name
        final = evals[-1]
        assert final['iteration'] == manifest['iters'], name
        assert len(final['cases']) == 40, name
        assert all(c['score'] is not None for c in final['cases']), name
        count = sum(r['timeout_count'] for r in rows)
        results.append(dict(name=name, seed=config['seed'], mode=config['timeout_bootstrap'],
            final_score=final['score'], random_score=final['eval_random_score'],
            kickoff_score=final['eval_kickoff_score'], best_iteration=best,
            selected_score=next(r['score'] for r in evals if r['iteration']==best),
            timeout_count=count, timeout_value_mean=(sum(r['timeout_count']*r['timeout_bootstrap_mean']
                 for r in rows if r['timeout_count'])/count if count else None),
            final_cases=final['cases']))
    pairs = []
    for seed in manifest['seeds']:
        names = [f'{arm}_seed{seed}' for arm in ('B0','E1')]
        if not all(name in manifest['completed'] for name in names):
            continue
        states = [torch.load(root/name/'initial_policy.pt', map_location='cpu', weights_only=True) for name in names]
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(states[0][k], states[1][k]) for k in states[0]), names
        initial = [read_jsonl(root/name/'evaluations.jsonl')[0] for name in names]
        assert initial[0]['cases'] == initial[1]['cases'], names
        checks.append(f'Seed {seed}: pesos iniciais e painel inicial idênticos; orçamento verificado.')
        a,b = [next(r for r in results if r['name']==name) for name in names]
        pairs.append(dict(seed=seed, **{key:b[key]-a[key] for key in
                     ('final_score','random_score','kickoff_score')}))
    aggregate = {key:dict(mean=statistics.mean(p[key] for p in pairs),
                         min=min(p[key] for p in pairs), max=max(p[key] for p in pairs))
                 for key in ('final_score','random_score','kickoff_score')} if pairs else {}
    (root/'summary.json').write_text(json.dumps(dict(status=manifest['status'],results=results,
                               paired_deltas=pairs,aggregate=aggregate,checks=checks),indent=2))
    lines=['# Validação B0/E1', '', f"Status: {manifest['status']}. Braços concluídos: {len(results)}/{2*len(manifest['seeds'])}.", '',
           'Comparação principal: endpoint de orçamento fixo. Melhor checkpoint é resultado secundário',
           'selecionado no mesmo painel, sem validação independente.', '',
           '| Braço | Score final | Random | Kickoff | Melhor it. | Timeouts |',
           '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['name']} | {r['final_score']:.5f} | {r['random_score']:.5f} | {r['kickoff_score']:.5f} | {r['best_iteration']} | {r['timeout_count']} |")
    lines += ['', '## Diferenças pareadas E1 − B0', '', '| Seed | Score | Random | Kickoff |', '|---|---:|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['final_score']:+.5f} | {p['random_score']:+.5f} | {p['kickoff_score']:+.5f} |")
    lines += ['', *checks, '', 'Três seeds oferecem evidência exploratória. Casos, episódios e ambientes não são',
              'réplicas independentes de treinamento. O painel é greedy e tem resets determinísticos',
              'no kickoff: múltiplos ambientes podem repetir a mesma trajetória.',
              'Nenhum checkpoint é promovido automaticamente. Resultados completos por adversário,',
              'lado e seed estão em summary.json e nos evaluation_cases.jsonl de cada braço.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    summarize(Path(sys.argv[1]).resolve())
