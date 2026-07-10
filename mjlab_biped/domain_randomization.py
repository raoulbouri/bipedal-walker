"""
Domain randomization (DR) for the biped RL environment (Phase 6.C).

Per-episode multiplicative randomization of body mass/inertia, geom
friction, actuator kp/kv, and a discrete Jetson on/off toggle on the
torso body — applied to a compiled `mujoco.MjModel` in place, per the
frozen spec in `docs/events_spec.md`.

Every `apply()` call resamples from the PRISTINE baseline arrays
captured once at construction time — never from the model's current
(possibly already-randomized) live arrays. This guarantees repeated
resets are non-compounding: `apply(seed_A)` then `apply(seed_B)` must
match a fresh `DomainRandomizer` + `apply(seed_B)` bit-for-bit.

Pure Python + numpy + mujoco (no mjlab/torch imports — Colab-only per
CLAUDE.md, not in Mac core deps).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import mujoco

# Torso body name (composite housing + motor controller [+ Jetson]).
_TORSO_BODY_NAME = "composite_part_1__1_"

# Frozen Jetson on/off constants, matching scripts/postprocess.py's
# TORSO_BASE / TORSO_BASE+JETSON pair (kept in sync manually here, per
# docs/events_spec.md's explicit design decision to avoid a runtime
# dependency from mjlab_biped on scripts/).
_TORSO_MASS_WITH_JETSON = 0.279
_TORSO_INERTIA_WITH_JETSON = np.array([6e-4, 5e-4, 4e-4], dtype=np.float64)
_TORSO_MASS_NO_JETSON = 0.099
_TORSO_INERTIA_NO_JETSON = np.array([2.1e-4, 1.8e-4, 1.4e-4], dtype=np.float64)


@dataclass
class DomainRandomizationCfg:
    """Domain randomization ranges and toggles."""

    enabled: bool = False
    mass_range: Tuple[float, float] = (0.9, 1.1)
    friction_range: Tuple[float, float] = (0.7, 1.3)
    gain_range: Tuple[float, float] = (0.8, 1.2)
    jetson_toggle_enabled: bool = True


class DomainRandomizer:
    """
    Captures a pristine baseline of a model's DR-relevant arrays at
    construction time, and applies fresh multiplicative randomization to
    the model in place on each `apply()` call.
    """

    def __init__(self, model: mujoco.MjModel):
        self._nbody = model.nbody
        self._nu = model.nu

        self._baseline_body_mass = model.body_mass.copy()
        self._baseline_body_inertia = model.body_inertia.copy()
        self._baseline_geom_friction = model.geom_friction.copy()
        self._baseline_actuator_gainprm = model.actuator_gainprm.copy()
        self._baseline_actuator_biasprm = model.actuator_biasprm.copy()

        self._torso_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, _TORSO_BODY_NAME
        )
        if self._torso_body_id < 0:
            raise ValueError(f"body '{_TORSO_BODY_NAME}' not found in model")

        self._baseline_torso_mass = float(model.body_mass[self._torso_body_id])
        self._baseline_torso_inertia = model.body_inertia[self._torso_body_id].copy()

    def apply(
        self,
        model: mujoco.MjModel,
        rng: np.random.Generator,
        cfg: DomainRandomizationCfg,
    ) -> None:
        """Mutate `model` in place per `cfg`, always resampling from the
        pristine baseline captured in `__init__` (never from the model's
        current live state)."""
        if not cfg.enabled:
            model.body_mass[:] = self._baseline_body_mass
            model.body_inertia[:] = self._baseline_body_inertia
            model.geom_friction[:] = self._baseline_geom_friction
            model.actuator_gainprm[:] = self._baseline_actuator_gainprm
            model.actuator_biasprm[:] = self._baseline_actuator_biasprm
            return

        # --- Body mass + inertia (independent factor per body, i >= 1) ---
        mass_lo, mass_hi = cfg.mass_range
        for i in range(1, self._nbody):
            factor = rng.uniform(mass_lo, mass_hi)
            model.body_mass[i] = self._baseline_body_mass[i] * factor
            model.body_inertia[i] = self._baseline_body_inertia[i] * factor

        # --- Friction (one shared factor across all nonzero-friction geoms) ---
        fric_lo, fric_hi = cfg.friction_range
        friction_factor = rng.uniform(fric_lo, fric_hi)
        nonzero_mask = self._baseline_geom_friction[:, 0] != 0.0
        model.geom_friction[nonzero_mask, 0] = (
            self._baseline_geom_friction[nonzero_mask, 0] * friction_factor
        )

        # --- Actuator gains (independent kp, kv factor per actuator) ---
        gain_lo, gain_hi = cfg.gain_range
        for i in range(self._nu):
            kp_factor = rng.uniform(gain_lo, gain_hi)
            kv_factor = rng.uniform(gain_lo, gain_hi)
            model.actuator_gainprm[i, 0] = self._baseline_actuator_gainprm[i, 0] * kp_factor
            model.actuator_biasprm[i, 1] = self._baseline_actuator_biasprm[i, 1] * kp_factor
            model.actuator_biasprm[i, 2] = self._baseline_actuator_biasprm[i, 2] * kv_factor

        # --- Jetson on/off toggle (torso body mass + diagonal inertia) ---
        if cfg.jetson_toggle_enabled:
            jetson_on = bool(rng.choice([True, False]))
            if jetson_on:
                model.body_mass[self._torso_body_id] = _TORSO_MASS_WITH_JETSON
                model.body_inertia[self._torso_body_id] = _TORSO_INERTIA_WITH_JETSON
            else:
                model.body_mass[self._torso_body_id] = _TORSO_MASS_NO_JETSON
                model.body_inertia[self._torso_body_id] = _TORSO_INERTIA_NO_JETSON
