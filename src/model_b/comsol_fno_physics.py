"""
The PCNO code referenced in this file is by Z. Xue, "A robust physics-constrained
neural operator framework for efficient geothermal resource development" [source
code], Code Ocean, 2026, https://doi.org/10.24433/CO.2608907.v1 (GPL-3.0).

3D physics-informed loss module: mass + energy conservation residuals,
adapted from PCNO's own validated physics (PCNO_Model_4D.py's `PDE_F` and
`physical_loss`) to our native 3D grid (60x30x20, real Z-extent -- not the
degenerate single-Z-layer trick the earlier 2D PINN track needed) and to
this dataset's actual COMSOL well-control scheme.

`PDE_F` below is copied verbatim from PCNO_Model_4D.py (pure density/
viscosity/enthalpy/upstream-weighting/harmonic-mean math, no adaptation
needed) -- reusing validated physics rather than re-deriving it, same
principle the project's earlier 2D PINN track followed (pcno_compat.py).

`MassEnergyConservationLoss` is an ADAPTED version of PCNO's
`physical_loss` -- NOT a verbatim reuse, because two real, evidence-based
differences exist between PCNO's own dataset and ours (both verified
directly against the actual COMSOL build scripts, not assumed):

1. PRODUCER WELL TERM. PCNO's original `phy_loss()` derives the
   producer's flow rate from a bottom-hole pressure via a Peaceman
   well-index formula: `q_out = I/mu * (pwf - p_block) * rho` -- i.e.
   PCNO's own dataset uses PRESSURE-CONTROLLED producers. Our dataset's
   `template_light.m` shows BOTH wells use `wellInputType='MassFlow'`,
   and both reference the SAME rate variable `m`
   (`well1.set('M0','m')`, `well2.set('M0','m')`) -- a balanced,
   RATE-CONTROLLED injector/producer doublet. There is no `pwf` in this
   dataset at all. Feeding `pwf=0` would silently zero the entire
   producer term (PCNO's own code masks on `pwf_local != 0`) while
   looking like it trained fine -- exactly the kind of bug this project
   has caught before by verifying against real data, not just code
   review. Fixed here: the producer is treated symmetrically to the
   injector (a directly specified rate sink), using `inj_rate` at the
   producer's `well_map` cell -- matching what this dataset's COMSOL
   simulations actually do.

2. LAYER THICKNESS (dz). PCNO's `physical_loss` hardcodes
   `self.dz = dz / 5.0` (their own dataset uses 5 Z-layers). Our grid
   has 20 Z-layers with a real, uniform physical spacing available
   directly from the prepared dataset's `z_m` coordinates (~21.05m
   between adjacent layers). Fixed here: `dz` is passed in already
   computed from `z_m`, with no /5.0 division and no resampling needed.

Verified parameter/unit mapping (from the actual COMSOL build scripts,
not assumed):
- `inj_rate` is directly in kg/s (`build_models_fno_lhs.m`:
  `setParamIf(..., 'inj_rate', 'm', @(v) sprintf('%.6g[kg/s]', v))`) --
  PCNO's own volumetric-to-mass `*1001.7/24/3600` conversion does NOT
  apply here; used directly.
- `PDE_F.ent()`'s formula (`t = (T+273.15)/1000`) expects degrees C, not
  K -- use `inj_temp_C`, and field predictions must be converted from
  our stored Kelvin (`T_K - 273.15`) before being passed in.
- `depth`: not a per-scenario parameter in this dataset (not among the
  14 scalar params, not sampled in the COMSOL build script). Confirmed
  via `template_light.m`: `model.param.set('TD','-2500[m]-H/2')` -- a
  fixed 2500m base depth plus half the (per-scenario) thickness `H`.
  The prepared dataset's `z_m` coordinates (`z_m[0]=-2700m`,
  spacing~21.05m) are consistent with this and give full per-cell
  resolution; used here as a representative scalar (`mean(|z_m|)`) to
  match `physical_loss`'s existing scalar-depth usage pattern.
- Rock thermal conductivity (`lam_r`): confirmed NOT a per-scenario
  parameter -- COMSOL uses a fixed temperature-dependent function
  (`template_light.m`, material `mat4`'s Piecewise `k(T)`). Evaluated
  at a representative reservoir temperature (~400K) this gives
  ~2.86 W/(m*K), a typical reservoir-rock value -- used as a fixed
  constant (`ROCK_THERMAL_CONDUCTIVITY` below), not per-scenario data
  we don't have. Cross-checked: the same template's fluid (water)
  conductivity function evaluates to ~0.687 W/(m*K) at the same
  temperature, close to PCNO's own hardcoded fluid-conductivity
  constant (0.648, from `5.6e4/86400`) -- good evidence the two
  codebases' physics assumptions are compatible.

Deliberately NOT ported (yet): PCNO's `WellboreModel` and the
wellhead-temperature/task-variable losses built on top of it. That
model converts tensors to plain Python floats and calls `iapws`,
severing the autograd graph (documented already in this project's
earlier 2D PINN work, `pcno_lite/physics.py`) -- it's also the most
complex piece and needs its own dedicated adaptation. `PDE_F` itself
does NOT use `iapws` at all (self-contained polynomial fits), so this
first phase (mass + energy conservation only) needs no new dependency
installed on ARC.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_DEBUG = False  # set True (or monkeypatch from a smoke test) to print intermediate NaN/range checks

# Representative constant -- see module docstring. Not a per-scenario
# input; COMSOL uses a fixed temperature-dependent function for this,
# not something this dataset's LHS design varies.
ROCK_THERMAL_CONDUCTIVITY = 2.86  # W/(m*K)


# ---------------------------------------------------------------------------
# PDE_F -- copied verbatim from PCNO_Model_4D.py. Pure density/viscosity/
# enthalpy/upstream-weighting/harmonic-mean math, dimension-agnostic (works
# on any X/Y/Z grid size), no adaptation needed for our 3D grid.
# ---------------------------------------------------------------------------
class PDE_F:
    """
    Static physics-utility namespace holding PCNO's water density,
    viscosity, and enthalpy correlations, plus the harmonic-mean
    permeability and upstream-weighting helpers that
    MassEnergyConservationLoss composes into the Darcy mass/energy
    residuals. Every method is a @staticmethod operating only on its
    arguments -- the class is never instantiated and holds no instance
    state.
    """
    @staticmethod
    def den(p, T):
        """
        Computes liquid water density from an exponential
        compressibility/thermal-expansion correlation referenced to
        100 kPa, 25 C.

        Args:
          p: Pressure, kPa.
          T: Temperature, degrees C.
        Returns:
          Density, kg/m^3.
        """
        den_0 = 1000.11  # kg/m3
        a = 7e-7  # kPa-1
        b = 8.8e-4  # C-1
        pr = 100  # kPa
        Tr = 25  # C
        return den_0 * torch.exp(a * (p - pr) - b * (T - Tr))

    @staticmethod
    def vis(T):
        """
        NaN-gradient fix (2026-07-27): val2's base (1 + 0.015512*(T-20)) can
        go negative for T < ~-44.5C, and a negative base raised to a
        fractional power (-1.572) is NaN. Before this fix, T_pred was
        always exactly time-invariant (see comsol_fno_model.py's
        time-conditioning fix), so the network had already converged to
        physically sensible temperatures before physics_weight ramped up --
        this branch's NaN was never actually reached. Once T_pred genuinely
        varies over time, an unconverged network early in training can
        predict a cell colder than -44.5C; even though torch.where's outer
        cond1 (T<40) correctly selects val1 for that cell's FORWARD value,
        val2's internal NaN still corrupts the BACKWARD gradient at that
        exact position (torch.where computes each branch's local gradient
        everywhere, and 0*NaN=NaN, not 0 -- confirmed directly: an isolated
        extreme T produced a NaN gradient only at that one position, with
        neighboring normal-T positions staying finite). That single NaN
        entry then poisons the model's shared weights (reused across every
        cell/timestep) once backprop reaches them, and NaN never recovers --
        matching the observed failure: a real ARC run (2026-07-27) using
        the time-fixed architecture went NaN in every loss (not just the
        physics ones) starting exactly at the epoch physics_weight first
        reached 0.15. Fixed by clamping val2's base to stay positive before
        the power -- val2 is only physically used for 40<=T<100C anyway
        (cond2), so this only changes behavior on already-unphysical input,
        not the valid operating range.
        """
        cond1 = T < 40
        cond2 = (T >= 40) & (T < 100)
        val1 = 1.787 * torch.exp(T * (-0.033 + 0.0001962 * T))
        val2 = 1e-3 * torch.clamp(1 + 0.015512 * (T - 20), min=1e-6) ** -1.572
        val3 = 0.02414 * 10 ** (247.8 / (T + 133.15))
        result = torch.where(cond1, val1, torch.where(cond2, val2, val3))
        return result * 0.001

    @staticmethod
    def ent(T):
        """
        Computes the specific enthalpy of liquid water from a quartic
        polynomial correlation in reduced temperature (t = (T+273.15)/1000).

        Args:
          T: Temperature, degrees C.
        Returns:
          Specific enthalpy, J/kg.
        """
        A = -203.6060
        B = 1523.29
        C = -3196.413
        D = 2474.455
        E = 3.855326
        Fc = -256.5478
        H_const = -285.8304
        t = (T + 273.15) / 1000
        H_val = A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + Fc - H_const
        return (H_val * 1000) / 0.01802

    @staticmethod
    def harmonic_mean_x(k, eps=1e-30):
        """
        Computes the harmonic mean of permeability `k` between each
        interior cell and its left/right neighbor along the X axis, used
        as the inter-cell transmissibility for the Darcy flux term. Also
        crops the Y/Z axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          k:   (B, X, Y, Z, T) permeability field, m^2.
          eps: Small constant preventing division by zero where both
               neighbors have near-zero permeability.
        Returns:
          (mean_k_left, mean_k_right), each cropped to the interior grid.
        """
        k_l, k_c, k_r = k[:, :-2, :, :, :], k[:, 1:-1, :, :, :], k[:, 2:, :, :, :]
        mean_k_l = 2.0 * k_c * k_l / (k_c + k_l + eps)
        mean_k_r = 2.0 * k_c * k_r / (k_c + k_r + eps)
        return mean_k_l[:, :, 1:-1, 1:-1, :], mean_k_r[:, :, 1:-1, 1:-1, :]

    @staticmethod
    def harmonic_mean_y(k, eps=1e-30):
        """
        Computes the harmonic mean of permeability `k` between each
        interior cell and its neighbor on either side along the Y axis,
        used as the inter-cell transmissibility for the Darcy flux term.
        Also crops the X/Z axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          k:   (B, X, Y, Z, T) permeability field, m^2.
          eps: Small constant preventing division by zero where both
               neighbors have near-zero permeability.
        Returns:
          (mean_k_d, mean_k_u), each cropped to the interior grid.
        """
        k_d, k_c, k_u = k[:, :, :-2, :, :], k[:, :, 1:-1, :, :], k[:, :, 2:, :, :]
        mean_k_d = 2.0 * k_c * k_d / (k_c + k_d + eps)
        mean_k_u = 2.0 * k_c * k_u / (k_c + k_u + eps)
        return mean_k_d[:, 1:-1, :, 1:-1, :], mean_k_u[:, 1:-1, :, 1:-1, :]

    @staticmethod
    def harmonic_mean_z(k, eps=1e-30):
        """
        Computes the harmonic mean of permeability `k` between each
        interior cell and its neighbor on either side along the Z axis,
        used as the inter-cell transmissibility for the Darcy flux term.
        Also crops the X/Y axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          k:   (B, X, Y, Z, T) permeability field, m^2.
          eps: Small constant preventing division by zero where both
               neighbors have near-zero permeability.
        Returns:
          (mean_k_back, mean_k_front), each cropped to the interior grid.
        """
        k_b, k_c, k_f = k[:, :, :, :-2, :], k[:, :, :, 1:-1, :], k[:, :, :, 2:, :]
        mean_k_b = 2.0 * k_c * k_b / (k_c + k_b + eps)
        mean_k_f = 2.0 * k_c * k_f / (k_c + k_f + eps)
        return mean_k_b[:, 1:-1, 1:-1, :, :], mean_k_f[:, 1:-1, 1:-1, :, :]

    @staticmethod
    def upstream_x(x, p):
        """
        Upstream-weights field `x` at each cell face along the X axis: the
        value carried across a face is taken from whichever of the two
        neighboring cells has the higher pressure `p` (the direction fluid
        actually flows from), which stabilizes the advection term. Also
        crops the Y/Z axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          x: (B, X, Y, Z, T) advected quantity (e.g. density, viscosity,
             enthalpy) at cell centers.
          p: (B, X, Y, Z, T) pressure field used to determine flow
             direction, Pa.
        Returns:
          (x_right, x_left), upstream-weighted values at the right/left
          faces, each cropped to the interior grid.
        """
        p_l, p_c, p_r = p[:, :-2, :, :, :], p[:, 1:-1, :, :, :], p[:, 2:, :, :, :]
        pd_r = (p_r - p_c >= 0).float()
        x_r = x[:, 1:-1, :, :, :] * (1.0 - pd_r) + x[:, 2:, :, :, :] * pd_r
        x_r = x_r[:, :, 1:-1, 1:-1, :]
        pd_l = (p_c - p_l >= 0).float()
        x_l = x[:, :-2, :, :, :] * (1.0 - pd_l) + x[:, 1:-1, :, :, :] * pd_l
        x_l = x_l[:, :, 1:-1, 1:-1, :]
        return x_r, x_l

    @staticmethod
    def upstream_y(x, p):
        """
        Upstream-weights field `x` at each cell face along the Y axis: the
        value carried across a face is taken from whichever of the two
        neighboring cells has the higher pressure `p` (the direction fluid
        actually flows from), which stabilizes the advection term. Also
        crops the X/Z axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          x: (B, X, Y, Z, T) advected quantity (e.g. density, viscosity,
             enthalpy) at cell centers.
          p: (B, X, Y, Z, T) pressure field used to determine flow
             direction, Pa.
        Returns:
          (x_u, x_d), upstream-weighted values at the two Y faces, each
          cropped to the interior grid.
        """
        p_d, p_c, p_u = p[:, :, 2:, :, :], p[:, :, 1:-1, :, :], p[:, :, :-2, :, :]
        pd_u = (p_u - p_c >= 0).float()
        x_u = x[:, :, 1:-1, :, :] * (1.0 - pd_u) + x[:, :, :-2, :, :] * pd_u
        x_u = x_u[:, 1:-1, :, 1:-1, :]
        pd_d = (p_d - p_c >= 0).float()
        x_d = x[:, :, 1:-1, :, :] * (1.0 - pd_d) + x[:, :, 2:, :, :] * pd_d
        x_d = x_d[:, 1:-1, :, 1:-1, :]
        return x_u, x_d

    @staticmethod
    def upstream_z(x, p):
        """
        Upstream-weights field `x` at each cell face along the Z axis: the
        value carried across a face is taken from whichever of the two
        neighboring cells has the higher pressure `p` (the direction fluid
        actually flows from), which stabilizes the advection term. Also
        crops the X/Y axes to the interior grid used elsewhere in
        MassEnergyConservationLoss.

        Args:
          x: (B, X, Y, Z, T) advected quantity (e.g. density, viscosity,
             enthalpy) at cell centers.
          p: (B, X, Y, Z, T) pressure field used to determine flow
             direction, Pa.
        Returns:
          (x_f, x_b), upstream-weighted values at the front/back Z faces,
          each cropped to the interior grid.
        """
        p_b, p_c, p_f = p[:, :, :, :-2, :], p[:, :, :, 1:-1, :], p[:, :, :, 2:, :]
        pd_f = (p_f - p_c >= 0).float()
        x_f = x[:, :, :, 1:-1, :] * (1.0 - pd_f) + x[:, :, :, 2:, :] * pd_f
        x_f = x_f[:, 1:-1, 1:-1, :, :]
        pd_b = (p_b - p_c >= 0).float()
        x_b = x[:, :, :, 1:-1, :] * (1.0 - pd_b) + x[:, :, :, :-2, :] * pd_b
        x_b = x_b[:, 1:-1, 1:-1, :, :]
        return x_f, x_b


# ---------------------------------------------------------------------------
# Adapted mass + energy conservation loss (see module docstring for the two
# real adaptations: direct-rate producer, un-divided dz).
# ---------------------------------------------------------------------------
class MassEnergyConservationLoss:
    """
    All spatial tensors (Cp_r, dz, q_inj_field, T_inj_field, k, phi,
    prod_rate_field) are (B, X, Y, Z, 1) -- broadcast across the time
    axis. p_pred/T_pred passed to __call__ are (B, X, Y, Z, T), T in
    degrees C, p in MPa (matching PCNO's own convention, verified in
    PCNO_Model_4D.py: `p_Pa = p_pred * 1e6`).
    """

    def __init__(self, Cp_r, dz, q_inj_field, T_inj_field, prod_rate_field, k, phi, dx, dy, dt):
        """
        Stores this batch's per-cell static physics fields and grid
        spacings, and derives the fluid/rock pore-volume split and the
        fluid-rock effective thermal conductivity used by the mass and
        energy residuals computed in __call__.

        Args:
          Cp_r:            (B, X, Y, Z, 1) rock specific heat, J/(kg*K).
          dz:              (B, 1, 1, 1, 1) layer thickness, m.
          q_inj_field:     (B, X, Y, Z, 1) injector mass-rate field, kg/s
                            (nonzero only at injector cells).
          T_inj_field:     (B, X, Y, Z, 1) injection temperature field,
                            degrees C (nonzero only at injector cells).
          prod_rate_field: (B, X, Y, Z, 1) producer mass-rate magnitude
                            field, kg/s (nonzero only at producer cells).
          k:               (B, X, Y, Z, 1) permeability, m^2.
          phi:             (B, X, Y, Z, 1) porosity, dimensionless.
          dx, dy:          Cell dimensions in the X/Y directions, m.
          dt:              Time between consecutive field snapshots, s.
        """
        self.dx, self.dy, self.dt = float(dx), float(dy), float(dt)
        self.Cp_r = Cp_r
        self.dz = dz  # NOT divided by 5 -- see module docstring, point 2
        self.q_inj_field = q_inj_field
        self.T_inj_field = T_inj_field
        self.prod_rate_field = prod_rate_field  # positive magnitude; sign applied internally
        self.k = k
        # NOTE: PCNO_Model_4D.py applies k_unit_conversion=9.86923e-16 (mD->m^2)
        # because their own raw k input is in mD. Our dataset's log10_perm
        # decodes to raw values ~1e-13 (e.g. -13 to -12.6), which is already a
        # physically sensible m^2 value (~100-250 mD-equivalent) -- NOT mD
        # itself (1e-13 mD would be absurdly, unphysically tiny). So no
        # conversion is applied here; k is used as-is (already m^2).
        self.k_unit_conversion = 1.0

        self.lam_fluid = 5.6e4 / (24 * 3600)
        self.lam_rock = ROCK_THERMAL_CONDUCTIVITY / (24 * 3600)
        self.lam_eff = self.lam_fluid * ((self.lam_rock / self.lam_fluid) ** 0.6354)

        block_v = dx * dy * self.dz
        self.fluid_v = block_v * phi
        self.rock_v = block_v - self.fluid_v

    @staticmethod
    def _pad_3d(x, mode='replicate'):
        """
        Pads a (B, X, Y, Z, T) tensor by 1 cell on each side of the X, Y,
        and Z axes (the time axis is left untouched), so the finite-
        difference stencils in __call__ have a neighbor to read at every
        cell, including those on the physical grid boundary.

        Args:
          x:    (B, X, Y, Z, T) tensor to pad.
          mode: Padding mode passed to F.pad (default 'replicate', i.e.
                edge cells are repeated outward).
        Returns:
          (B, X+2, Y+2, Z+2, T) padded tensor.
        """
        x = x.permute(0, 4, 3, 2, 1)
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode=mode)
        return x.permute(0, 4, 3, 2, 1)

    def __call__(self, p_pred, T_pred, valid_mask=None):
        """
        valid_mask: (B, X, Y, Z, T) bool/float, typically mask_T & mask_P
        combined -- True/1 where BOTH fields have real (non-cap-rock,
        non-placeholder) data. Cap-rock cells are placeholder-trained, not
        real physics, so their "residual" is meaningless and would
        otherwise inject spurious gradient signal. A residual between two
        consecutive timesteps is only counted where BOTH the earlier and
        later timestep's cell is valid, matching what mass_acc/energy_acc
        actually depend on. If None, no masking is applied (only
        appropriate if valid_mask is already guaranteed 100% valid, e.g.
        a toy all-finite test case).
        """
        device = p_pred.device
        S_c = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1), slice(1, None))

        # k/q_inj_field/T_inj_field/prod_rate_field are time-invariant,
        # stored with a size-1 broadcast last dim. But S_c (and the
        # `[..., 1:]` slicing throughout this method) slices the time axis
        # explicitly -- on a real size-1 dim that produces an EMPTY tensor,
        # not a broadcast one (this is exactly the same class of bug fixed
        # above for fluid_v/rock_v/Cp_r, just requiring the opposite fix
        # here since these particular fields genuinely do get time-sliced
        # by S_c/[...,1:] elsewhere in this method). Expand to the real
        # timestep count first so the slicing has real elements to act on.
        T = p_pred.shape[-1]

        def expand_t(x):
            """Broadcasts a time-invariant (B, X, Y, Z, 1) field out to
            the batch's real timestep count `T`, so it can be sliced by
            the same time-axis indexing used elsewhere in __call__."""
            return x.expand(-1, -1, -1, -1, T)

        p_pred = self._pad_3d(p_pred)
        T_pred = self._pad_3d(T_pred)
        k_pad = self._pad_3d(expand_t(self.k).to(device))
        q_inj_pad = self._pad_3d(expand_t(self.q_inj_field).to(device))
        T_inj_pad = self._pad_3d(expand_t(self.T_inj_field).to(device))
        prod_rate_pad = self._pad_3d(expand_t(self.prod_rate_field).to(device))

        p_Pa = p_pred * 1e6
        rho = torch.clamp(PDE_F.den(p_Pa / 1000.0, T_pred), 500.0, 2000.0)
        mu = torch.clamp(PDE_F.vis(T_pred), 1e-4, 1e1)
        h = torch.nan_to_num(PDE_F.ent(T_pred), nan=0.0, posinf=1e7, neginf=-1e7).clamp(-1e6, 1e6)
        k = (k_pad * self.k_unit_conversion).to(device)

        if _DEBUG:
            for name, t in [('p_pred', p_pred), ('T_pred', T_pred), ('rho', rho), ('mu', mu), ('h', h), ('k', k)]:
                print(f'  [{name}] nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} '
                      f'min={t.min().item():.4g} max={t.max().item():.4g}')

        T_c, p_c, rho_c, mu_c, h_c, k_c = T_pred[S_c], p_Pa[S_c], rho[S_c], mu[S_c], h[S_c], k[S_c]
        p_t, rho_t, mu_t, h_t, k_t = p_Pa[..., 1:], rho[..., 1:], mu[..., 1:], h[..., 1:], k[..., 1:]
        dz_b = self.dz.to(device=k_c.device, dtype=k_c.dtype).view(-1, 1, 1, 1, 1)

        # Injector: directly specified rate source (matches PCNO's own treatment).
        q_inj = q_inj_pad[S_c]
        H_inj = PDE_F.ent(T_inj_pad)[S_c]
        energy_inj = q_inj * H_inj

        # Producer: ADAPTED -- directly specified rate sink (negative), not
        # derived from a bottom-hole pressure. See module docstring, point 1.
        q_out = -prod_rate_pad[S_c]
        energy_out = q_out * h_c

        I = (2.0 * torch.pi) * k_c * dz_b  # kept for parity with PCNO's well-index term shape; unused in the rate-controlled formulation
        del I

        rho_r, rho_l = PDE_F.upstream_x(rho_t, p_t)
        rho_u, rho_d = PDE_F.upstream_y(rho_t, p_t)
        rho_f, rho_b = PDE_F.upstream_z(rho_t, p_t)
        mu_r, mu_l = PDE_F.upstream_x(mu_t, p_t)
        mu_u, mu_d = PDE_F.upstream_y(mu_t, p_t)
        mu_f, mu_b = PDE_F.upstream_z(mu_t, p_t)
        h_r, h_l = PDE_F.upstream_x(h_t, p_t)
        h_u, h_d = PDE_F.upstream_y(h_t, p_t)
        h_f, h_b = PDE_F.upstream_z(h_t, p_t)
        k_l, k_r = PDE_F.harmonic_mean_x(k_t)
        k_d, k_u = PDE_F.harmonic_mean_y(k_t)
        k_b, k_f = PDE_F.harmonic_mean_z(k_t)

        if _DEBUG:
            for name, t in [('rho_r', rho_r), ('mu_r', mu_r), ('h_r', h_r), ('k_l', k_l), ('k_r', k_r),
                             ('k_d', k_d), ('k_u', k_u), ('k_b', k_b), ('k_f', k_f)]:
                print(f'  [{name}] nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} '
                      f'min={t.min().item():.4g} max={t.max().item():.4g} shape={tuple(t.shape)}')

        rho_next, rho_last = rho[:, 1:-1, 1:-1, 1:-1, 1:], rho[:, 1:-1, 1:-1, 1:-1, :-1]
        T_next, T_last = T_pred[:, 1:-1, 1:-1, 1:-1, 1:], T_pred[:, 1:-1, 1:-1, 1:-1, :-1]
        p_next, p_last = p_Pa[:, 1:-1, 1:-1, 1:-1, 1:], p_Pa[:, 1:-1, 1:-1, 1:-1, :-1]
        h_next, h_last = h[:, 1:-1, 1:-1, 1:-1, 1:], h[:, 1:-1, 1:-1, 1:-1, :-1]

        p_r, p_l = p_Pa[:, 2:, 1:-1, 1:-1, 1:], p_Pa[:, 0:-2, 1:-1, 1:-1, 1:]
        p_u, p_d = p_Pa[:, 1:-1, 0:-2, 1:-1, 1:], p_Pa[:, 1:-1, 2:, 1:-1, 1:]
        p_b, p_f = p_Pa[:, 1:-1, 1:-1, 0:-2, 1:], p_Pa[:, 1:-1, 1:-1, 2:, 1:]
        T_r, T_l = T_pred[:, 2:, 1:-1, 1:-1, 1:], T_pred[:, 0:-2, 1:-1, 1:-1, 1:]
        T_u, T_d = T_pred[:, 1:-1, 0:-2, 1:-1, 1:], T_pred[:, 1:-1, 2:, 1:-1, 1:]
        T_b, T_f = T_pred[:, 1:-1, 1:-1, 0:-2, 1:], T_pred[:, 1:-1, 1:-1, 2:, 1:]

        Ax, Ay, Az = self.dy * dz_b, self.dx * dz_b, self.dx * self.dy

        tx_r = (rho_r / mu_r) * k_r * (Ax / self.dx)
        tx_l = (rho_l / mu_l) * k_l * (Ax / self.dx)
        ty_u = (rho_u / mu_u) * k_u * (Ay / self.dy)
        ty_d = (rho_d / mu_d) * k_d * (Ay / self.dy)
        tz_b = (rho_b / mu_b) * k_b * (Az / self.dz)
        tz_f = (rho_f / mu_f) * k_f * (Az / self.dz)

        mass_flux = (tx_r * (p_r - p_c) + tx_l * (p_l - p_c) + ty_u * (p_u - p_c) + ty_d * (p_d - p_c) +
                     tz_f * (p_f - p_c) + tz_b * (p_b - p_c))
        # fluid_v/rock_v/Cp_r are static per-cell quantities, (B,X,Y,Z,1) --
        # broadcast naturally against the (...,T-1)-shaped time-diffed
        # tensors below via their size-1 last dim. NOT sliced by time (an
        # earlier version incorrectly did `[..., 1:]` here, which on a
        # size-1 dim produces an empty tensor, not a broadcastable one --
        # matches PCNO's own code, which never slices these either).
        fluid_v = self.fluid_v.to(device)
        rock_v = self.rock_v.to(device)
        Cp_r = self.Cp_r.to(device)

        # Residual mask: a transition between two consecutive timesteps only
        # counts where BOTH the earlier and later timestep's cell is real
        # data (not cap-rock/placeholder) -- see __call__'s docstring.
        # mass_residual/energy_residual are (B,X,Y,Z,T-1), matching the
        # ORIGINAL (unpadded) grid resolution -- valid_mask must be at that
        # same resolution already (it is: mask_T/mask_P from the dataset).
        if valid_mask is not None:
            res_mask = (valid_mask[..., 1:] * valid_mask[..., :-1]).to(device=device, dtype=torch.float32)
            res_mask_sum = res_mask.sum().clamp(min=1.0)

            def masked_mean_abs(residual):
                """
                Computes the mean absolute residual over cells where
                `res_mask` marks both the earlier and later timestep as
                real (non-placeholder) data.

                Args:
                  residual: (B, X, Y, Z, T-1) mass or energy residual.
                Returns:
                  Scalar mean absolute residual over valid cells.
                """
                return (torch.abs(residual) * res_mask).sum() / res_mask_sum
        else:
            def masked_mean_abs(residual):
                """
                Computes the mean absolute residual over every cell,
                unweighted -- used when no valid_mask is supplied.

                Args:
                  residual: (B, X, Y, Z, T-1) mass or energy residual.
                Returns:
                  Scalar mean absolute residual.
                """
                return torch.mean(torch.abs(residual))

        mass_acc = (rho_next - rho_last) * fluid_v / self.dt
        mass_residual = mass_acc - mass_flux - (q_inj + q_out)
        # *5e-1 matches PCNO_Model_4D.py's own empirically-tuned scaling,
        # carried over as a reasonable starting point (real magnitude
        # check done via smoke test against actual data, not guessed) --
        # may need retuning once mixed into the staged training loss.
        mse_mass = masked_mean_abs(mass_residual) * 5e-1

        energy_acc = ((fluid_v * rho_next * (h_next - p_next / rho_next) +
                       rock_v * (Cp_r * T_next)) -
                      (fluid_v * rho_last * (h_last - p_last / rho_last) +
                       rock_v * (Cp_r * T_last))) / self.dt

        lam_r_ = (rho_r / mu_r) * h_r * k_r * (Ax / self.dx)
        lam_l_ = (rho_l / mu_l) * h_l * k_l * (Ax / self.dx)
        lam_u_ = (rho_u / mu_u) * h_u * k_u * (Ay / self.dy)
        lam_d_ = (rho_d / mu_d) * h_d * k_d * (Ay / self.dy)
        lam_b_ = (rho_b / mu_b) * h_b * k_b * (Az / self.dz)
        lam_f_ = (rho_f / mu_f) * h_f * k_f * (Az / self.dz)

        energy_conv = (lam_r_ * (p_r - p_c) + lam_l_ * (p_l - p_c) + lam_u_ * (p_u - p_c) + lam_d_ * (p_d - p_c) +
                       lam_f_ * (p_f - p_c) + lam_b_ * (p_b - p_c))
        energy_cond = self.lam_eff * ((T_r - T_c + T_l - T_c) * (Ax / self.dx) +
                                       (T_u - T_c + T_d - T_c) * (Ay / self.dy) +
                                       (T_f - T_c + T_b - T_c) * (Az / self.dz))

        energy_residual = energy_acc - energy_conv - energy_cond - (energy_inj - energy_out)
        # *1e-8 matches PCNO_Model_4D.py's own empirically-tuned scaling --
        # same rationale as mse_mass's *5e-1 above.
        mse_energy = masked_mean_abs(energy_residual) * 1e-8

        return mse_mass, mse_energy


def erode_mask_3d(mask):
    """
    Shrinks a (B,X,Y,Z,T) validity mask by 1 cell in every X/Y/Z direction --
    a cell only survives erosion if it AND all 6 face-adjacent neighbors are
    valid. Needed because MassEnergyConservationLoss's finite-difference
    stencils read neighboring cells: a cell sitting right at the reservoir/
    cap-rock boundary could have a technically-valid mask value itself, but
    its residual would still be computed FROM a neighbor's placeholder-
    filled value, contaminating it. Masking only the final average (without
    this erosion) doesn't catch that -- callers should erode before passing
    a mask into MassEnergyConservationLoss.__call__.
    """
    inv = 1.0 - mask.float()
    # channels-last (B,X,Y,Z,T) -> (B,T,X,Y,Z) for a spatial-only 3D max-pool
    inv = inv.permute(0, 4, 1, 2, 3)
    any_invalid_neighbor = F.max_pool3d(inv, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - any_invalid_neighbor
    return eroded.permute(0, 2, 3, 4, 1)


# ---------------------------------------------------------------------------
# Input decoder: maps our prepared-dataset channels/params into what
# MassEnergyConservationLoss expects, in the right units.
# ---------------------------------------------------------------------------
def build_physics_loss(x_static_raw, params_raw, static_channel_names, param_names,
                        z_m, x_m, y_m, dt_seconds):
    """
    x_static_raw: (B, 8, X, Y, Z) RAW (not normalized) static channels.
    params_raw:   (B, 14) RAW (not normalized) scalar params.
    z_m, x_m, y_m: 1D coordinate arrays (numpy) from the prepared dataset.
    dt_seconds:   seconds between the field snapshots this loss will be
                  evaluated on consecutive pairs of (matches
                  comsol_fno_train.py's step=5-year grid -> 5*365*24*3600).

    Returns a MassEnergyConservationLoss instance for this batch.

    NOTE: does not need initial-condition fields (T_i/p_i in PCNO's own
    constructor) -- those were only used by the wellbore-model section of
    PCNO's physical_loss, which this mass+energy-only adaptation doesn't
    include (see module docstring).
    """
    name_to_idx = {n: i for i, n in enumerate(static_channel_names)}
    param_to_idx = {n: i for i, n in enumerate(param_names)}

    def static(name):
        """Looks up one raw static channel by name.

        Args:
          name: Channel name (a key of `static_channel_names`).
        Returns:
          (B, X, Y, Z) tensor for that channel, in raw (not normalized)
          physical units.
        """
        return x_static_raw[:, name_to_idx[name]]  # (B, X, Y, Z)

    def param(name):
        """Looks up one raw scalar parameter by name.

        Args:
          name: Parameter name (a key of `param_names`).
        Returns:
          (B,) tensor for that parameter, in raw (not normalized) physical
          units.
        """
        return params_raw[:, param_to_idx[name]]  # (B,)

    device = x_static_raw.device
    B, _, X, Y, Z = x_static_raw.shape

    phi = static('phi').unsqueeze(-1)  # (B,X,Y,Z,1)
    k = (10.0 ** static('log10_perm')).unsqueeze(-1)  # raw perm, m^2 (verified via magnitude check -- see module docstring; k_unit_conversion=1.0 accordingly)
    well_map = static('well_map')  # (B,X,Y,Z), +1 injector / -1 producer

    inj_rate = param('inj_rate').view(B, 1, 1, 1, 1)  # kg/s, already correct unit -- see module docstring
    inj_temp_c = param('inj_temp_C').view(B, 1, 1, 1, 1)
    cp_rock = param('cp_rock').view(B, 1, 1, 1, 1).expand(B, X, Y, Z, 1)

    inj_mask = (well_map > 0.5).unsqueeze(-1).float()
    prod_mask = (well_map < -0.5).unsqueeze(-1).float()

    q_inj_field = inj_rate * inj_mask
    T_inj_field = inj_temp_c * inj_mask  # PDE_F.ent only evaluated where nonzero-relevant; zero elsewhere is fine (multiplied by q_inj=0 there)
    prod_rate_field = inj_rate * prod_mask  # rate-controlled doublet: producer rate == injector rate (see module docstring)

    dz = float(abs(z_m[1] - z_m[0]))
    dz_t = torch.full((B, 1, 1, 1, 1), dz, device=device, dtype=x_static_raw.dtype)
    dx = float(abs(x_m[1] - x_m[0]))
    dy = float(abs(y_m[1] - y_m[0]))

    Cp_r = cp_rock

    return MassEnergyConservationLoss(
        Cp_r=Cp_r, dz=dz_t, q_inj_field=q_inj_field, T_inj_field=T_inj_field,
        prod_rate_field=prod_rate_field, k=k, phi=phi, dx=dx, dy=dy, dt=dt_seconds,
    )
