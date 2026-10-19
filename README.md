# Impostor_Agent

# DeceptiveUAV


<video src="ICRA-VIDEO_final_try7.mp4" controls width="100%"></video>

A deceptive agent **D** inside a multi-UAV swarm fabricates obstacle reports to
delay honest agents from reaching their goals, while evading a Bayesian
trust-fusion detector. This repo holds the simulator, the attackers (learned
and rule-based), the baselines, the evaluation harness, and a Gazebo /
Crazyflie visualisation and deployment path.

The metric throughout is **delay**: the increase in mission makespan caused by
D's presence, measured against the identical grid with D's slot removed.

The model weights are present in this link:
* [Model Weights](https://drive.google.com/drive/folders/1igEK3sLVWKhg35LjVMuie6Xs_GXtWeqi?usp=drive_link)

---

## Contents

1. [Quick start](#1-quick-start)
2. [Environment](#2-environment)
3. [Trust and detection](#3-trust-and-detection)
4. [Attackers](#4-attackers)
5. [Training pipeline](#5-training-pipeline)
6. [Sensitivity sweeps](#8-sensitivity-sweeps)
7. [Crazyflie / Gazebo](#10-crazyflie--gazebo)


---

## 1. Quick start

```bash
# a 50-seed benchmark
python3 -m experiments.eval_szvoren          
```

---

## 2. Environment

`core/env.py`. Every evaluation uses the same construction:

```python
grid, rng = create_grid(M=45, density=0.15, obstacle_type='maze', seed=seed)
starts    = place_points(grid, 5, rng, 0.3)
targets   = place_points(grid, 5, rng, 0.3, min_sep=12, existing=starts)
full_grid, base_grid, extra_cells = add_obstacles_on_top(
    grid, extra_density=0.05, seed=seed, starts=starts, targets=targets)
assignment = assign_targets_labelled(starts, targets, full_grid)
```

| parameter | value | meaning |
|---|---|---|
| `M` | 45 | grid side (45×45 cells) |
| `density` | 0.15 | base-grid obstacle fraction |
| `obstacle_type` | `'maze'` | generator; also `clustered`, `random` |
| `extra_density` | 0.05 | **add-on-top** obstacles, unknown to everyone at t=0 |
| `n_agents` | 5 | 4 honest + 1 deceptive |
| `min_sep` (targets) | 12 | minimum target separation, cells |
| `margin` | 0.3 / 0.15 | placement clearance / planning clearance |
| `exclusion_radius` | 2.0 | add-on-top cells kept clear of starts/targets |
| `seeds` | `range(200000, 200050)` | the 50 evaluation seeds |


### Simulation loop

`sim/loop.py::run_swarm`.

| parameter | eval value meaning |
|---|---|---|---|
| `T_comm` | 5 | steps between broadcast rounds |
| `r_sense` | 5.0 | sensing radius, cells |
| `v_max` | 1.0 | cells per step |
| `max_steps` | 300 | episode cap (**censors delay** — see §11) |
| `planner` | `'astar' | also `rrt_star` |
| `sample_planning` | `False` | deterministic threshold, not stochastic |
| `stuck_noise_prob` | 0.10 | safety-valve probability (below) |
| `dynamic_trust` | `True` | run the trust signals |
| `target_mode` | `'labelled'`| fixed agent→target assignment |
| `fab_mode` | `'all'` | D may fabricate anywhere legal |


**`stuck_noise_prob`** (`sim/planning.py::plan_with_stuck_noise`) is a safety
valve, not noise for its own sake. A well-placed lie in `sample_planning=False`
mode can leave an agent with *no path at all* — no movement, no re-sensing,
permanently locked. With the valve, each failed replan has a 0.10 chance of
re-planning on a grid with 25% of the agent's *unverified believed* obstacles
dropped. Catastrophic locks degrade into long-but-finite delays. Personally
sensed cells are never dropped.

### Baseline for delay

```python
rb = run_swarm(..., remove_agent_id=0)      # correct
```

`remove_agent_id=0` deletes D's slot — start, target **and** assignment — so the
baseline is exactly `n_honest` agents with `n_honest` targets.

---

## 3. Trust and detection

`core/constants.py::TrustParams`, `trust/signals.py`, `trust/beliefs.py`.

### Belief fusion

Each honest agent maintains `prob_belief_map`, P(obstacle) per cell, by
log-odds accumulation:

| cell class | log-odds | P | note |
|---|---|---|---|
| personally sensed | ±∞ | 0 or 1 | hard ground truth, overrides everything |
| base-grid obstacle | +4.0 | 0.98 | strong prior, not absolute |
| base-grid free | **−0.7** | 0.33 | **weak** — one trusted report can flip it |
| unreported, unexplored | — | 0.0 | optimistic free |
| global prior `prior_p` | — | 0.15 | used where no base value applies |

Each sender contributes `log(τ/(1−τ))` to the cells it reports, where τ is the
receiver's trust in it. **Senders below τ = 0.35 are skipped entirely**, not
inverted — a distrusted agent is unreliable, not an anti-oracle.

The planner thresholds at **P > 0.5** (`planning_grid`). Consequence worth
knowing: on a base-free cell, a *single* report only crosses 0.5 once
**τ ≥ 0.668** (weight 0.70 must overcome the −0.7 prior). Since initial trust is
0.70, D sits 0.03 above the belief threshold — any trust dip disables all of its
claims at once, and any full recovery re-enables them at once.

### Trust signals

| # | signal | parameter | value |
|---|---|---|---|
| 1 | retro-verification: confirm | `alpha_c` | 0.05 × (1−τ) |
| 1 | retro-verification: contradict | `alpha_x` | 0.15 × τ (**uncapped**) |
| 2 | direct contradiction | `alpha_x` | 0.15 |
| 3 | behavioural consistency | `gamma` | 0.03 |
| 4 | staleness decay | `delta`, `lambda_d` | 0.01, 0.05 |
| 5 | consensus disagreement | `zeta` | 0.02 |

| other | value |
|---|---|
| `FROZEN_TRUST` (initial) | 0.70 |
| `trust_floor` / `trust_ceiling` | 0.05 / 0.99 |
| `staleness_threshold` | 3 rounds |
| `consistency_ema_rate` | 0.2 |
| `near/far_target_threshold` | 0.2 / 0.5 |
| `near_target_distance` | 2.0 |
| `obstacle_discovery_grace` | 5 steps |
| `alert_threshold` | 0.40 |
| `alert_epsilon` | 0.10 |
| `alert_accused_weight` | 0.5 |
| `alert_accuser_penalty` | 0.15 |
| `MAX_RETRO_BOOST_PER_ROUND` | 0.05 (**see §11**) |

### Detection

```python
quorum = max(1, ceil(n_honest * detection_quorum_fraction))   # fraction = 1/3
detected = (n_agents_with_trust_below_0.35 >= quorum)
```

| n_honest | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| quorum | 1 | 1 | **2** | 2 | 2 |

The step function matters when sweeping agent count — detection difficulty jumps
between 3 and 4 honest agents and is flat thereafter. Detection is **recorded,
not terminal**: the episode runs to completion so makespan and detection step
are independent measurements.

---

## 4. Attackers

Selected by `d_agent_type` in `run_swarm`. All share a 16-cell-per-round
fabrication budget.

| `d_agent_type` | class | scoring | paper role |
|---|---|---|---|
| `heuristic` | `adversary/heuristic.py` | geometric | weak rule-based reference |
| `greedy` | `adversary/greedy.py` | forked simulation | strong rule-based |
| `obstacle_attack` | `adversary/baseline_obstacle_attack.py` | A\* or fork | **Szvoren et al. ICRA 2025** |
| `nn` | `adversary/policy.py` | learned + optional fork | **ours** |

### Szvoren et al. baseline — `obstacle_attack`

Faithful port of the ICRA 2025 brute-force obstacle attack (Algorithm 1):
exhaustively try an obstacle at every cell along the victim's current A\* path,
keep the placement maximising path length.

```python
d_agent_kwargs = {'budget_cells': 16, 'obstacles_per_round': 1,
                  'victim': 'all', 'one_shot': False, 'scoring': 'astar'}
```

| parameter | default | meaning |
|---|---|---|
| `budget_cells` | 16 | cells per round |
| `obstacles_per_round` | 1 | → `obstacle_size = √(16/1) = 4`, one 4×4 block |
| `victim` | `'all'` | pool candidates from every active agent; `'longest'` = literal single-robot attack |
| `one_shot` | `False` | recompute each round; `True` reproduces the paper |
| `scoring` | `'astar'` | published method; `'fork'` ablation; `'random'` control |
| `fork_top_k` | 8 | candidates forked in `'fork'` mode |
| `allow_unreachable` | `True` | see below |

Budget geometry — only perfect squares spend the full budget:

```
opr= 1: 1 × 4x4 = 16 cells   ✓      opr= 2: 2 × 2x2 =  8 cells   ✗ half wasted
opr= 4: 4 × 2x2 = 16 cells   ✓      opr=16: 16 × 1x1 = 16 cells  ✓
```

**Adaptations from the paper**, all switchable:

1. *Single robot → makespan.* `_score` takes `max()` over victims, since makespan
   is set by the last arrival.
2. *"One obstacle" → 16 cells.* The paper's obstacle is a physical object with
   extent; mapping it to one grid cell would be a strawman.
3. *One-shot → per round.* Gives the baseline the same number of opportunities
   as ours.
4. *Knowledge restricted.* Uses `_build_known_grid()` (base + own sensing +
   received reports), **not** the full map the paper grants its adversary. This
   makes the baseline slightly weaker than published — state it when reporting.
5. *Detection-naive by construction.* The paper has no trust model, so this
   attacker never reasons about being caught.
6. *`allow_unreachable=True`.* The paper forbids fully blocking the robot
   because that means a permanent lock. Here it doesn't — `plan_with_stuck_noise`
   exists for exactly that case — so blocking is a legal and effective move.
   Measured: a single 4×4 block strands an agent in 0/781 candidate evaluations,
   but 16 separate 1×1 cells do so 15.3% of the time.

### Greedy — `greedy`

```python
{'strategy': 'by_delay', 'budget_per_round': 16, 'n_candidates': 10,
 'lookahead_steps': 100, 'full_grid': ..., 'base_grid': ...}
```

`while remaining_budget > 0`: generate path-closure and chokepoint **wall
segments**, score each by `simulate_counterfactual`, commit the best, repeat.
`strategy` ∈ `truly_greedy` (furthest agent) / `by_delay` (all agents) /
`hybrid` (lock onto round-1 winner). `selection='random'` picks uniformly from
the same candidates — this is the `results/random.txt` control.

### Greedy vs Szvoren

Same objective and (in fork mode) the same evaluator. They differ in:

- **candidate shape** — k×k square blocks on path cells vs. wall *segments*
  perpendicular to corridors at chokepoints;
- **budget structure** — a fixed count of fixed-size blocks vs.
  `while remaining_budget > 0` composing several complementary segments per
  round, re-solving A\* between each.

Those two account for the +14.4 → +55.8 gap; the scoring function does not.


### Ours — `nn`

```python
{'model_path': MODEL, 'inference_selector': 'q',
 'inference_fork_verify_topk': 3, 'inference_closure_repair': False}
```

| parameter | default | meaning |
|---|---|---|
| `inference_n_samples` | 8 | candidate walls decoded per round |
| `inference_temperature` | 1.0 | sampling temperature |
| `inference_selector` | `'astar'` | ranking: `'astar'` proxy or `'q'` (Q-head) |
| `inference_fork_verify_topk` | 0 | 0 = simulator-free; k>0 forks the top k |
| `inference_closure_repair` | `False` | repair pass on near-closures |
| `closure_repair_budget` | 4 | cells for that pass |
| `target_return` | 2000.0 | RvS conditioning value at inference |
| `deterministic` | `False` | argmax vs sampling |
| `use_plan_context` | `True` | feed plan latent / round history |


## 5. Training pipeline

Three stages, same network throughout (`models/policy_net.py`).

### Architecture

| parameter | value |
|---|---|
| `EMBED_DIM` | 256 |
| `N_HEADS` | 8 |
| `N_LAYERS` | 6 |
| `PATCH_POOL` | 8 (→ 64 patches) |
| `DROPOUT` | 0.1 |
| `MAX_AGENTS` | 10 |
| `MAX_CELLS_PER_ROUND` | 16 |
| `SCALAR_DIM` | 7 (7th = `target_return`) |
| input channels | 8 (occupancy, belief, exploration, D position, honest footprint, recency, lie-age) |
| output | cell-index sequence + STOP token |

`MAX_AGENTS=10` with an `agent_mask` means **variable agent counts need no
retraining** — 2 to 6 honest agents all work with the same checkpoint.

### Stage 1 — behaviour cloning (`train/bc.py::pretrain`)

```
n_epochs=50, batch_size=32, lr=1e-4, episode_chaining=True
```
Sequence NLL on the oracle's cell sequence + value MSE on the score, weighted by
demonstration quality.

### Stage 2 — RvS (`train/rvs.py::rcrl_finetune`)

```
n_epochs=20, batch_size=32, lr=5e-5, episode_chaining=True
RETURN_TARGET_MIN=0.0, RETURN_TARGET_MAX=280.0
```

**This is return-conditioned supervised learning (RvS, Emmons et al. ICLR 2022),
not Decision Transformer.** The return enters as a single normalised scalar in a
7-dim feature vector — `scalars[6]` — and the model learns `π(a | s, ω)`. There
is no causal transformer over `(R̂, s, a)` trajectory tokens. The action decoder
*is* autoregressive, but over cells within one wall, which is a different axis.
The function's own docstring says "RvS / Decision-Transformer style"; that
conflates the two. Cite Emmons et al.

`episode_chaining=False` reverts to plain i.i.d. batching and leaves
`plan_head` / `round_history_encoder` without gradient.

### Stage 3 — PPO (`train/ppo.py::ppo_finetune`)

| parameter | value |
|---|---|
| `n_episodes` | 1000 |
| `max_steps_per_ep` | 300 |
| `lr` | 3e-5 |
| `entropy_coeff` | 0.02 |
| `kl_coeff` | 0.15 (anchor to the RvS policy) |
| `buffer_size` | 128 |
| `inference_target_return` | 2000.0 |
| `vary_target_return` | `True` |
| `eval_every` / `eval_n_seeds` / `eval_seed_base` | 50 / 30 / 100000 |
| `eval_fork_verify_topk` | **0** |
| `n_baseline_runs` | 3 |
| `q_baseline_warmup_updates` | 20 |
| `collapse_window` / `fab` / `delay` | 30 / 5.0 / 3.0 |
| `max_revivals` | 3 |


## 6. Sensitivity sweeps

```bash
python3 -m experiments.sensitivity 1      # honest agent count: 2,3,5,6
python3 -m experiments.sensitivity 2      # add-on-top density: 0.05 ... 0.40
python3 -m experiments.sensitivity 3      # stuck-noise prob:   0.05 ... 0.30
python3 experiments/analyze_sensitivity.py 1
```

Writes `results/sensitivity_test{N}.csv`, one row per (condition, seed), with
both the D makespan and its matched baseline.

### Test 1 — honest agent count

| honest | quorum | base_mk | mk | delay | delay % | det | det_step | stalled |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 47.5 | 108.1 | +60.6 | +124.8% | 88% | 73.2 | 10% |
| 3 | 1 | 51.5 | 120.9 | +69.4 | +129.1% | 86% | 64.8 | 12% |
| 4 | **2** | 54.7 | 173.6 | +118.9 | **+212.8%** | 74% | **112.7** | 26% |
| 5 | 2 | 57.8 | 146.0 | +88.3 | +149.8% | 76% | 77.0 | 20% |
| 6 | 2 | 59.3 | 142.7 | +83.3 | +139.3% | 84% | 84.4 | 16% |

An inverted U peaking at 4. The quorum step (1→2 at honest=4) explains it, and
`det_step` confirms: D survives longest exactly where delay peaks. By 6 agents,
sensor density claws detection back to 84%. Note `base_mk` rises on its own
(47.5 → 59.3) because makespan is the *last* arrival — compare delay %, not raw
delay.

### Test 2 — add-on-top density

| density | base_mk | mk | delay | delay % | det | stalled |
|---|---|---|---|---|---|---|
| 0.05 | 54.7 | 173.6 | +118.9 | +212.8% | 74% | 26% |
| 0.10 | 61.9 | 171.5 | +109.5 | +174.9% | 62% | 30% |
| 0.15 | 74.5 | 201.2 | +126.7 | +174.3% | 64% | 34% |
| 0.20 | 100.0 | 228.2 | +128.2 | +138.3% | 56% | 52% |
| 0.25 | 137.3 | 226.5 | +89.1 | +75.8% | 66% | 40% |
| 0.30 | 184.6 | 264.4 | +79.8 | +49.1% | 58% | 52% |
| 0.35 | 199.4 | 270.2 | +70.8 | +41.4% | 60% | 62% |
| 0.40 | 199.0 | 283.4 | +84.4 | +51.6% | 57% | 71% |

Delay % falls monotonically; raw delay stays flat (+70 to +128) while the
baseline quadruples. The environment does the delaying at high density, leaving
D less headroom. **Cap the reported sweep at 0.25** — beyond that, 52–71% of
episodes are censored.

---


## 7. Crazyflie / Gazebo

```bash
bash crazyflie/run_seed_gazebo.sh 200001                     # simulate + launch + play
bash crazyflie/run_seed_gazebo.sh 200001 --paused            # hold frame 0 (recording)
bash crazyflie/run_seed_gazebo.sh 200001 --view-agent 3      # UAV 3's belief map
bash crazyflie/run_seed_gazebo.sh 200001 --view-agent 0      # D's own map
bash crazyflie/run_seed_gazebo.sh 200001 --regen --speed 2
```

Camera and playback, from a second terminal:

```bash
python3 -m crazyflie.gazebo_view --workspace <ws.json> --view top --zoom 1.4
python3 -m crazyflie.gazebo_ctl pause | play | rewind | restart | speed 2 | seek 40 | quit
```

`rewind` returns to frame 0 **keeping** pause state (set up a shot); `restart`
rewinds and plays.

Use `venv/bin/python3` — the system `python3` has no numpy.
