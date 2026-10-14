"""GAE for goal-terminal episodes with optional artificial time-limit bootstrap."""
import torch


@torch.no_grad()
def gae_returns(rewards, values, terminated, done, final_values, last_value,
                gamma, lam, timeout_bootstrap=True):
    """All rollout tensors are [time, learner]; last_value is [learner].

    final_values contains pre-autoreset values for timeout-only transitions.
    Goals (including simultaneous goal + timeout) always have zero bootstrap.
    The trace stops at either boundary, including in legacy control mode.
    """
    advantages = torch.zeros_like(rewards)
    trace = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        ordinary = last_value if t == len(rewards) - 1 else values[t + 1]
        successor = torch.where(done[t], final_values[t], ordinary)
        boundary = terminated[t] if timeout_bootstrap else done[t]
        bootstrap = torch.where(boundary, torch.zeros_like(successor), successor)
        delta = rewards[t] + gamma * bootstrap - values[t]
        trace = delta + gamma * lam * torch.where(done[t], torch.zeros_like(trace), trace)
        advantages[t] = trace
    return advantages, advantages + values
