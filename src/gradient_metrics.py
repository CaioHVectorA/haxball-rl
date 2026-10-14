"""Objective-specific diagnostics; autograd.grad leaves optimizer gradients untouched."""
import torch


def objective_gradient_metrics(model, actor_loss, critic_loss):
    params = tuple(model.parameters())
    actor = torch.autograd.grad(actor_loss, params, retain_graph=True, allow_unused=True)
    critic = torch.autograd.grad(critic_loss, params, retain_graph=True, allow_unused=True)
    def norm(grads):
        return torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None))
    trunk_ids = {id(p) for p in model.trunk.parameters()}
    pairs = [(a.detach(), c.detach()) for p, a, c in zip(params, actor, critic)
             if id(p) in trunk_ids and a is not None and c is not None]
    an = norm([a for a, _ in pairs])
    cn = norm([c for _, c in pairs])
    cosine = (float(sum((a*c).sum() for a, c in pairs)/(an*cn))
              if float(an*cn) > 0 else None)
    return dict(grad_norm_actor_pre_clip=float(norm(actor)),
                grad_norm_critic_pre_clip=float(norm(critic)), trunk_grad_cosine=cosine)
