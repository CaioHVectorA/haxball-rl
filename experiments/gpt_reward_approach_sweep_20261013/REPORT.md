# Self-play reward sweep: approach weight

## Question

Does lowering the player-to-ball shaping reward reduce the overly direct, defensive/chasing behavior seen in the live match?

## Setup

Six matched trials trained from the same iteration-650 policy. Each trial ran 300 PPO iterations with 256 parallel environments and a fresh optimizer/value head. The only reward change was the coefficient on `MotionPlayerToBall`; goal reward stayed at ±25 and `MotionBallToGoal` stayed at 0.08. Seeds 0 and 1 were used for each coefficient. The starting checkpoint SHA-256 was `96d037566a676e719ec10f1be5c159d719a3c77229b6cef24977dd31ba624465`.

The reported score averages completed matches from randomized starting positions against five opponent types, evaluated on both sides and two fixed evaluation seeds. It is win = 1, draw = 0.5, loss = 0. Kickoff matches are excluded from this comparison.

## Results

| Approach weight | Seed 0: initial → final | Seed 1: initial → final | Mean final | Mean change |
|---:|---:|---:|---:|---:|
| 0.05 | 62.2% → 69.8% (+7.6 pp) | 62.2% → 72.5% (+10.3 pp) | 71.2% | +9.0 pp |
| 0.15 | 62.2% → 67.4% (+5.1 pp) | 62.2% → 74.1% (+11.9 pp) | 70.8% | +8.5 pp |
| 0.60 | 62.2% → 62.4% (+0.2 pp) | 62.2% → 63.5% (+1.2 pp) | 63.0% | +0.7 pp |

Both lower weights improved the evaluated match score over two seeds; 0.05 and 0.15 are nearly tied. This supports lowering 0.60, but two training seeds are only a screening result and do not establish a universal best coefficient.

## Behavior check

A separate fixed-seed inspection against a positional keeper found near-contact kick presses in 1.4–3.3% of contacts for weights 0.05 and 0.15, versus 4.1–4.7% for 0.60. The fraction of near-contact presses made while positioned beyond the ball relative to the opponent goal was about 60% for 0.05, 26–27% for 0.15, and 68–71% for 0.60. This is an action-intent proxy, not proof of a successful or correctly aimed shot. It suggests 0.15 is the better next candidate for testing the corner-angle problem, even though its match score is essentially tied with 0.05.

The inspection used one side and one fixed state seed. It did not reproduce the exact wall-corner situation from the live room, and goals are not attributed to the last toucher. The live room remains on checkpoint `it650`; none of these trial checkpoints was promoted.

## Next experiment

Use 0.15 as the provisional anchor and add a focused evaluation with the ball near the lower and upper walls, a stationary opponent, and several player approach angles. Measure repositioning, contact, ball direction after a kick, and completed match outcomes before adding an angle-shaping term to the production reward.
