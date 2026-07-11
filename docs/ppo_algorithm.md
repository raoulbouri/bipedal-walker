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
- **Actor observation `o_t`**: 24 raw deployment-available dims per frame
  (joint pos ×6, joint vel ×6, gyro ×3, previous action ×6, velocity
  command ×3), history-stacked over H=5 frames → **120-dim actor input**
  `x_t = [o_{t-4}, …, o_t]`.
  Removing orientation makes this a **POMDP**: gyro is angular *velocity*,
  so absolute tilt is not instantaneously observable — the stack gives the
  network a 100 ms window to integrate it. (CS285 handles partial
  observability by conditioning π on observation histories; stacking is
  the simplest instance.)
- **Critic observation `c_t`**: 39 privileged dims, single frame (actor's
  24 + projected gravity, foot touch, accelerometer, base linvel, base
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
| Actor π_θ | `x_t` (120) → EmpiricalNormalization | MLP [512, 256, 128], ELU | mean μ_θ(x_t) ∈ R^6 |
| Actor log-std | — | free parameter vector (state-independent), init σ=1.0 | σ ∈ R^6 |
| Critic V_φ | `c_t` (39) → EmpiricalNormalization | MLP [512, 256, 128], ELU | scalar V̂ |

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
  value — RSL-RL bootstraps it (reward is augmented with γ·V_φ(c_t) when
  `time_outs` is set). Conflating these teaches the policy that
  surviving to the time limit is as bad as falling. This is exactly why
  Phase 6.D kept `terminated` and `truncated` separate.
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
| Rollout | 120-dim stack (normalized) | 39-dim privileged (normalized) | sample a_t; record log-prob and V̂ |
| GAE | — | V̂ along trajectory | compute Â_t, R_t |
| Update | 120-dim stack | 39-dim privileged | recompute log π_θ, V_φ on minibatches; losses in §5 |
| Eval / deployment | 120-dim stack only | **never** | a = μ_θ(x), deterministic |

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

## CS285 lecture map

| Ingredient | CS285 |
|---|---|
| Policy gradient, causality/reward-to-go, baselines | Lecture 5 |
| Actor-critic, bootstrapping, bias–variance, GAE | Lecture 6 |
| Importance sampling, trust region, TRPO→PPO surrogate | Lecture 9 |
| POMDP → condition π on observation history | Lectures 2/5 remarks |
