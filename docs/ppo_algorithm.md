# PPO Training Algorithm Reference (Phase 7)

The exact algorithm RSL-RL's PPO runs on our task, written out so every
loss, target, and update is unambiguous. Framing follows Berkeley CS285
(Levine): policy gradients → causality/reward-to-go → baselines →
actor-critic with bootstrapping → GAE → trust-region-motivated surrogate
(PPO). Observation spec v1 (minimal-sensor actor) is assumed throughout —
see `docs/observation_spec.md` and CLAUDE.md Phase 7 v4.

## 1. Problem setup

- **True state `s_t`**: the full MuJoCo state (qpos, qvel, contacts).
  Exists only in sim; nothing is trained directly on it.
- **Actor observation `o_t`**: 28 raw deployment-available dims per frame
  (joint pos ×**8** — mjlab's built-in `joint_pos_rel` reports every hinge
  on the entity, i.e. the 6 actuated joints *and* the 2 passive ankles,
  not just the 6 you command — joint vel ×8 likewise, gyro ×3, previous
  action ×6, velocity command ×3), history-stacked over H=5 frames →
  **140-dim actor input**. Corrected 2026-07-13 from an earlier
  24/120 assumption that undercounted the joint terms — verified live
  against a real mjlab env, not re-derived from spec.
  **History stacking is per-TERM, not per-frame**: mjlab keeps a separate
  circular buffer per observation term and concatenates each term's own
  5-frame history, then concatenates terms — the real layout is
  `[pos_hist(40), vel_hist(40), gyro_hist(15), prevact_hist(30),
  cmd_hist(15)] = 140`, not `[frame_1(28), frame_2(28), …]`.
  Removing orientation makes this a **POMDP**: gyro is angular *velocity*,
  so absolute tilt is not instantaneously observable — the stack gives the
  network a 100 ms window to integrate it. (CS285 handles partial
  observability by conditioning π on observation histories; stacking is
  the simplest instance.)
  **`previous_action` carries a genuine one-step delay**: mjlab's
  `previous_action()` returns `action_manager.prev_action`, not
  `.action` — the frame at time `t` contains the action taken at `t-1`,
  not the action about to be taken. This is a real property of the
  training distribution (confirmed by tracing 10 steps of a live env),
  not an implementation artifact to normalize away when reconstructing
  or evaluating the observation elsewhere.
- **Critic observation `c_t`**: 43 privileged dims, single frame (actor's
  28 + projected gravity, foot touch, accelerometer, base linvel, base
  height, CoM). Near-Markovian, so no stacking. Training-only.
- **Action `a_t ∈ R^6`**: position targets for the 6 servos (kp=40, kv=10
  frozen in the model — the policy commands *where*, the servo model
  produces torque).
- **Reward `r_t`**: Phase 6.D terms (alive + upright − effort −
  action-rate; command-tracking zero-weighted for the balance gate).
  Computed from privileged sim state — legal, rewards never deploy.
- **Episode end**: `terminated` (fall: tilt or height) vs `truncated`
  (time-out). The distinction changes the value bootstrap (§4).
- Discount γ = 0.99, control at 50 Hz ⇒ effective horizon ≈ 1/(1−γ) =
  100 steps ≈ 2 s.

## 2. Networks

| Network | Input | Body | Output |
|---|---|---|---|
| Actor π_θ | `x_t` (140) → EmpiricalNormalization | MLP [512, 256, 128], ELU | mean μ_θ(x_t) ∈ R^6 |
| Actor log-std | — | free parameter vector (state-independent), init σ=1.0 | σ ∈ R^6 |
| Critic V_φ | `c_t` (43) → EmpiricalNormalization | MLP [512, 256, 128], ELU | scalar V̂ |

EmpiricalNormalization divides by `(std + eps)`, `eps=1e-2` (RSL-RL's
hardcoded default, not a configurable field anywhere in mjlab's config
surface, and not itself stored in the checkpoint — only `_mean`/`_var`/
`_std` are registered buffers). This matters in practice: `velocity_command`
is literally `[0,0,0]` in every frame of every episode until a real
`CommandTermCfg` is wired in, so its 15 history-stacked dims have exactly
zero variance in a trained checkpoint's own normalizer statistics —
normalizing with `(x-mean)/std` (no epsilon) divides by zero. Found this
the hard way reconstructing the observation outside mjlab for
`scripts/eval_checkpoint.py`: it produced a NaN action on the very first
inference call until the `+eps` was added.

Policy: diagonal Gaussian, π_θ(a|x) = N(μ_θ(x), diag(σ²)). During rollout
actions are **sampled** (exploration); at eval/deployment the **mean** is
used (deterministic). EmpiricalNormalization keeps running mean/var of
each obs dim and standardizes inputs; its statistics are saved in the
checkpoint and MUST be applied identically at deployment (Phase 7.D
parity test covers this).

## 3. Data collection (rollout)

Per iteration, N parallel envs (thousands, GPU) each step T=24 control
steps with the CURRENT policy:

store per step: `x_t`, `c_t`, `a_t`, `log π_θold(a_t|x_t)`, `r_t`,
`done_t`, `time_out_t`, `V_φ(c_t)`.

That's N×T transitions per iteration (e.g. 4096×24 ≈ 98k). PPO is
**on-policy**: this batch is used for one update phase, then discarded.

## 4. Targets ("ground truth") — GAE

The policy gradient is ∇J = E[Σ_t ∇log π(a_t|x_t) · Ψ_t]. Everything
below is about choosing Ψ_t well.

- **Causality / reward-to-go (CS285 Lec. 5):** action a_t cannot affect
  rewards already received, so Ψ_t sums only rewards from t onward:
  Ψ_t = Σ_{t'≥t} γ^{t'−t} r_{t'}. Dropping past rewards is unbiased and
  strictly reduces variance.
- **Baseline:** subtracting any state-dependent baseline b(s_t) leaves
  the gradient unbiased (E[∇log π · b] = 0) and reduces variance. The
  best practical baseline is the value function ⇒ the **advantage**
  A(s_t, a_t) = Q(s_t, a_t) − V(s_t): "how much better was this action
  than the policy's average from here."
- **Bootstrapped TD error (CS285 Lec. 6):**
  δ_t = r_t + γ·V_φ(c_{t+1})·(1 − terminated_t) − V_φ(c_t)
  is a one-step, low-variance (but biased, since V_φ is imperfect)
  advantage estimate.
- **GAE(λ):** exponentially-weighted blend of all n-step estimators,
  computed backward over each trajectory:
  A_t = δ_t + γλ·(1 − done_t)·A_{t+1},   λ = 0.95.
  λ→0 gives pure one-step TD (low variance, high bias); λ→1 gives pure
  Monte-Carlo reward-to-go minus baseline (unbiased, high variance);
  0.95 is the standard sweet spot.
- **Truncation vs termination bootstrap:** on a **fall** (true absorbing
  failure) the future value is genuinely 0 — no bootstrap. On a
  **time-out** the episode was cut artificially and the state still has
  value. Read the real mechanism directly in `rsl_rl/algorithms/ppo.py`
  rather than re-deriving it: the bootstrap is applied by **augmenting
  the reward at storage time**, before the GAE backward pass ever runs —
  `self.transition.rewards += self.gamma * (self.transition.values *
  extras["time_outs"])` — not by a separate branch inside the GAE
  recursion itself (which treats every `done` the same,
  `next_is_not_terminal=0`, whether it was a fall or a time-out; the
  reward already carries the correction by then). Conflating these
  (omitting the reward augmentation) teaches the policy that surviving
  to the time limit is as bad as falling — exactly why Phase 6.D kept
  `terminated` and `truncated` separate.
- **Value targets:** R_t = A_t + V_φ(c_t) (the TD(λ) return).
- **Advantage normalization:** per batch, A ← (A − mean)/(std + ε).
  Stabilizes the scale of the policy loss across iterations.

## 5. Losses

With ratio ρ_t(θ) = π_θ(a_t|x_t) / π_θold(a_t|x_t) (importance weight —
valid because the batch was collected under θ_old; CS285 Lec. 9):

- **Policy (clipped surrogate):**
  L^clip(θ) = −E_t[ min( ρ_t·Â_t,  clip(ρ_t, 1−ε, 1+ε)·Â_t ) ],  ε = 0.2.
  Why: the importance-sampled objective is only trustworthy near θ_old;
  TRPO enforces that with an explicit KL constraint, PPO approximates it
  by making the objective *pessimistic* once ρ leaves [1−ε, 1+ε] — the
  gradient through an out-of-range ratio (in the harmful direction) is
  zero, so repeated minibatch epochs on the same batch can't push the
  policy destructively far.
- **Value:** L^V(φ) = E_t[(V_φ(c_t) − R_t)²], with the *clipped* variant
  (`use_clipped_value_loss=True`): the value prediction is also clipped
  to within ε of its rollout-time prediction and the max of the two
  losses is taken — same trust-region logic applied to the critic.
- **Entropy bonus:** −c_ent·E[H[π_θ(·|x_t)]], c_ent = 0.005 — keeps σ
  from collapsing prematurely (exploration).
- **Total:** L = L^clip + c_V·L^V − c_ent·H,  c_V = 1.0.

## 6. Update mechanics

For each iteration's batch: 5 epochs × 4 minibatches; Adam; gradient-norm
clipped at 1.0. **Adaptive LR (RSL-RL's KL heuristic):** after each
minibatch, estimate KL(π_θold ‖ π_θ); if KL > 2×`desired_kl` (0.01) the
LR is divided by 1.5, if KL < `desired_kl`/2 it is multiplied by 1.5 —
an automatic step-size controller keeping updates inside an effective
trust region without TRPO's second-order machinery.

Then the updated θ becomes θ_old, a fresh rollout is collected, repeat.

## 7. What runs where (per data path)

| Stage | Actor input | Critic input | Uses |
|---|---|---|---|
| Rollout | 140-dim stack (normalized) | 43-dim privileged (normalized) | sample a_t; record log-prob and V̂ |
| GAE | — | V̂ along trajectory | compute Â_t, R_t |
| Update | 140-dim stack | 43-dim privileged | recompute log π_θ, V_φ on minibatches; losses in §5 |
| Eval / deployment | 140-dim stack only | **never** | a = μ_θ(x), deterministic |

## 8. Why these choices (one line each)

- **PPO over off-policy (SAC etc.):** thousands of parallel GPU envs make
  samples nearly free, so PPO's stability wins over sample efficiency.
- **Asymmetric critic:** the baseline may use any information without
  biasing the policy gradient; a better-informed V lowers advantage
  variance and never ships.
- **History stack over EKF:** no hand-built estimator to calibrate or
  transfer; the network learns whatever filter the task needs, from raw
  signals that exist on the real robot.
- **Gaussian with state-independent σ:** standard for locomotion;
  learned σ anneals naturally as the task is mastered.

## Where to read the real code (not a re-derivation)

- **Env-specific pieces** (observation/reward/action/termination
  definitions, exact real dims/ordering): `mjlab_biped/mjlab_task.py`.
- **Frozen hyperparameters**: `mjlab_biped/rl_cfg.py`.
- **The actual PPO engine** — GAE, both losses, KL-adaptive LR — lives in
  the installed `rsl_rl` package, not this repo:
  `rsl_rl/algorithms/ppo.py`'s `compute_returns()` (GAE backward pass,
  time-out reward augmentation) and `update()` (clipped surrogate, clipped
  value loss, entropy bonus, adaptive LR), plus
  `rsl_rl/storage/rollout_storage.py` (the minibatch generator). Reading
  this directly is worth it — it's genuine CS285-lecture-level code, not
  hidden behind abstraction.

## CS285 lecture map

| Ingredient | CS285 |
|---|---|
| Policy gradient, causality/reward-to-go, baselines | Lecture 5 |
| Actor-critic, bootstrapping, bias–variance, GAE | Lecture 6 |
| Importance sampling, trust region, TRPO→PPO surrogate | Lecture 9 |
| POMDP → condition π on observation history | Lectures 2/5 remarks |
