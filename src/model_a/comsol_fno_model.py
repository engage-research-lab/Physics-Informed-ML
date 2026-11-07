"""
Model architecture for the COMSOL/FNO 3D geothermal surrogate ("our model").

The PCNO code referenced in this file is by Z. Xue, "A robust physics-constrained
neural operator framework for efficient geothermal resource development" [source
code], Code Ocean, 2026, https://doi.org/10.24433/CO.2608907.v1 (GPL-3.0).

Heavily based on PCNO's own architecture (GPF + 3D U-Net + 4D FNO), adapted
to this project's grid (60x30x20 spatial, 9 uniform timesteps after the
step=5 resampling decision) and input channels (8 static spatial channels,
14 scalar params after dropping the constant well_radius).

Mirrors PCNO_Model_4D.py's real structure directly (SpectralConv4d's 8-corner
scheme, UNet3D_Lite's X/Y-only pooling, the gated UNet-inside-spectral-block
pattern) rather than a from-scratch design -- see
COMSOL_FNO_Dataset_Findings_and_Model_Plan.md and
Phase1_Implementation_and_Phase2_Plan.md for why. Padding/cropping constants
are computed from this project's actual shapes, not copied from PCNO's
(which were tuned to its different original grid).

Two separate single-field networks (one for T_K, one for p_MPa), matching
PCNO's own EnhancedP_T_Net design, rather than one shared multi-head model --
keeps the comparison against the PCNO baseline architecturally honest.

--- Two bug fixes applied 2026-07-27/28, both confirmed via a controlled
A/B test before being folded in here
as the canonical architecture -------------------------------------------
Fix 1 -- time-conditioning: FieldNet's input used to be exactly constant
across all 9 time snapshots, so SpectralConv4d -- which multiplies each
FFT frequency bin independently, never mixing bins -- could only ever
produce a time-constant output; fixed by appending a channel carrying the
normalized snapshot time, which actually varies across T.

Fix 2 -- GPF/GroupNorm interaction: GlobalParametricFusion (GPF) encodes
the 14 scalar params (inj_rate, inj_temp, well_spacing, geology-heterogeneity
stats, etc.) into a per-channel value broadcast uniformly across the whole
spatial grid, which then got concatenated with the spatial features and
immediately passed through `nn.GroupNorm(num_groups=8, num_channels=width)`.
Every configuration this project has ever trained uses width=8, so
num_groups==num_channels==8 -- GroupNorm collapses to per-channel INSTANCE
normalization, subtracting out each channel's own spatial mean. Since GPF's
contribution IS exactly that spatial mean (a broadcast constant), it was
being erased almost exactly, regardless of training. Confirmed empirically:
swapping in a completely different scenario's params (2.7 std devs) while
holding geology fixed changed the field prediction by ~1e-6 (floating-point
noise) with the un-fixed model; bypassing that one GroupNorm recovered a
real params-sensitivity signal (~0.08). Practical consequence: field-level
T_K/p_MPa predictions have never actually responded to injection
rate/temperature, well spacing, or any geology-statistics parameter -- only
to the spatially-resolved geology fields and well position.

Fixed via FiLM-style conditioning (Feature-wise Linear Modulation) -- GPF
now predicts a per-channel (gamma, beta) pair applied to the spatial branch
AFTER normalization, instead of being concatenated before the GroupNorm
that erases it. The old concat + fusion_conv1d path is removed since it's
no longer needed once GPF stops feeding through the shared tensor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SpectralConv4d -- PCNO's 8-corner hand-rolled 4D spectral convolution
# ---------------------------------------------------------------------------
class SpectralConv4d(nn.Module):
    """
    Same structure as PCNO_Model_4D.py's SpectralConv4d: FFT jointly over the
    last 4 dims (X, Y, Z, T), with 8 separate learned complex weight tensors
    -- one for every combination of low/high frequency band across X, Y, Z
    (2^3 = 8 corners). This is what makes the hand-rolled version more
    expressive than the neuraloperator library's single-region SpectralConv
    (confirmed in the earlier FNO-comparison track).
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        """
        in_channels/out_channels: channel counts this layer maps between.
        modes1/modes2/modes3/modes4: number of Fourier modes kept along X,
        Y, Z, T respectively; higher frequencies are truncated to zero.
        Allocates 8 separate complex weight tensors, one per low/high
        frequency-band combination across X, Y, Z (2^3 corners).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3, self.modes4 = modes1, modes2, modes3, modes4
        self.scale = 1 / (in_channels * out_channels)

        def cw():
            """Allocates one corner's randomly-initialized complex weight
            tensor, shape (in_channels, out_channels, modes1, modes2,
            modes3, modes4), scaled by 1/(in_channels*out_channels)."""
            return self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, modes4, dtype=torch.cfloat)

        self.weights = nn.ParameterList([nn.Parameter(cw()) for _ in range(8)])

    def compl_mul4d(self, x, weights):
        """Complex-valued 4D convolution in frequency space: contracts the
        input-channel dimension of `x` against `weights`' input/output
        channel dims, independently at every (X, Y, Z, T) frequency bin."""
        return torch.einsum("bixyzt,ioxyzt->boxyzt", x, weights)

    def forward(self, x):
        """
        x: (B, in_channels, X, Y, Z, T) real-valued.
        FFTs jointly over the last 4 dims, applies each of the 8 learned
        corner weight tensors to its corresponding low/high frequency band
        via compl_mul4d (all other frequencies stay zero), then inverse-FFTs
        back to real space.
        Returns: (B, out_channels, X, Y, Z, T).
        """
        B = x.shape[0]
        x = x.contiguous().to(dtype=torch.float32)
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])
        out_ft = torch.zeros(B, self.out_channels, x.size(-4), x.size(-3), x.size(-2),
                              x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        idx = lambda i: (slice(None), slice(None),
                          slice(None, self.modes1) if i & 1 == 0 else slice(-self.modes1, None),
                          slice(None, self.modes2) if i & 2 == 0 else slice(-self.modes2, None),
                          slice(None, self.modes3) if i & 4 == 0 else slice(-self.modes3, None),
                          slice(None, self.modes4))

        for i in range(8):
            out_ft[idx(i)] = self.compl_mul4d(x_ft[idx(i)], self.weights[i])

        return torch.fft.irfftn(out_ft, s=x.shape[-4:])


# ---------------------------------------------------------------------------
# 3D U-Net -- operates per-timestep (like PCNO's UNet3D_Lite), only pools X/Y
# ---------------------------------------------------------------------------
class UNet3D_Lite(nn.Module):
    """
    Same design as PCNO's UNet3D_Lite: two downsampling stages pooling only
    X and Y (kernel_size=(2,2,1)), a bottleneck, and matching upsampling
    stages with skip connections. Z is never pooled.
    """
    def __init__(self, in_channels, base_channels=16):
        """
        in_channels: number of input/output channels -- this U-Net maps
        in_channels -> in_channels (refines the feature map, doesn't
        change its channel count).
        base_channels: channel width of the first encoder stage; doubles
        at the second encoder stage and the bottleneck.
        """
        super().__init__()
        bc = base_channels

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, bc, 3, padding=1), nn.GroupNorm(4, bc), nn.ReLU(),
            nn.Conv3d(bc, bc, 3, padding=1), nn.GroupNorm(4, bc), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.enc2 = nn.Sequential(
            nn.Conv3d(bc, bc * 2, 3, padding=1), nn.GroupNorm(4, bc * 2), nn.ReLU(),
            nn.Conv3d(bc * 2, bc * 2, 3, padding=1), nn.GroupNorm(4, bc * 2), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.bottom = nn.Sequential(
            nn.Conv3d(bc * 2, bc * 4, 3, padding=1), nn.GroupNorm(4, bc * 4), nn.ReLU(),
            nn.Conv3d(bc * 4, bc * 2, 3, padding=1), nn.GroupNorm(4, bc * 2), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose3d(bc * 2, bc * 2, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec1 = nn.Sequential(nn.Conv3d(bc * 4, bc * 2, 3, padding=1), nn.GroupNorm(4, bc * 2), nn.ReLU())
        self.up2 = nn.ConvTranspose3d(bc * 2, bc, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.dec2 = nn.Sequential(nn.Conv3d(bc * 2, bc, 3, padding=1), nn.GroupNorm(4, bc), nn.ReLU())
        self.out_conv = nn.Conv3d(bc, in_channels, 1)

    def forward(self, x):
        """
        x: (B, in_channels, X, Y, Z) -- a single time snapshot (called once
        per timestep by Block4dWithUNet, not directly on the full 4D
        tensor). Two encoder stages pooling X/Y only, a bottleneck, then two
        decoder stages with skip connections back to the matching encoder
        stage.
        Returns: (B, in_channels, X, Y, Z), same spatial shape as the input.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottom(self.pool2(e2))
        d1 = self.dec1(torch.cat([e2, self._center_crop(self.up1(b), e2)], dim=1))
        d2 = self.dec2(torch.cat([e1, self._center_crop(self.up2(d1), e1)], dim=1))
        return self.out_conv(d2)

    @staticmethod
    def _center_crop(source, target):
        """Center-crops `source`'s spatial (H, W, D) dims down to match
        `target`'s, so an upsampled decoder feature map can be concatenated
        with its corresponding encoder skip connection."""
        _, _, h, w, d = target.shape
        sh, sw, sd = source.shape[2:]
        ch, cw, cd = (sh - h) // 2, (sw - w) // 2, (sd - d) // 2
        return source[:, :, ch:ch + h, cw:cw + w, cd:cd + d]


def compute_pad_for_pooling(size, n_pool_stages=2, pool_factor=2):
    """
    How much to pad a dimension (on one side only, for simplicity) so it
    divides evenly through `n_pool_stages` rounds of pooling by
    `pool_factor`. E.g. size=30, 2 stages of /2 -> needs divisibility by 4
    -> pad=2 (30+2=32, 32/4=8, clean).

    Computed from our actual grid (60x30x20), not copied from PCNO's own
    padding constants (which were tuned to its different 80x80x5 grid).
    """
    divisor = pool_factor ** n_pool_stages
    remainder = size % divisor
    return 0 if remainder == 0 else divisor - remainder


# ---------------------------------------------------------------------------
# Block4dWithUNet -- the core repeated block: spectral conv + pointwise
# linear + gated UNet contribution, matching PCNO's pattern
# ---------------------------------------------------------------------------
class Block4dWithUNet(nn.Module):
    """
    Repeated core block of the network: `n_layers` rounds of (spectral
    conv + pointwise 1x1 "linear" conv + a gated UNet3D_Lite contribution),
    summed, GroupNorm-normalized, and passed through GELU, followed by a
    final 2-layer MLP projecting from `width` channels down to `out_dim`.
    The UNet branch is computed once per forward pass (not once per layer)
    and mixed into every layer, scaled by a learned scalar `alpha`.
    """
    def __init__(self, width, width2, modes1, modes2, modes3, modes4, out_dim,
                 pad_x, pad_y, n_layers=4, unet_base_channels=None):
        """
        width: channel count carried through the spectral/UNet layers.
        width2: hidden width of the final pointwise MLP (fc1/fc2).
        modes1/modes2/modes3/modes4: Fourier modes kept along X, Y, Z, T,
        passed through to each SpectralConv4d layer.
        out_dim: number of output channels after the final MLP (1, for a
        single scalar field).
        pad_x/pad_y: pooling-divisibility padding already applied to the
        input along X/Y, cropped back off after the block's UNet
        pooling/upsampling round-trip.
        n_layers: number of stacked spectral+UNet layers.
        unet_base_channels: base channel width passed to UNet3D_Lite; if
        None, derived as max(4, width // 4).
        """
        super().__init__()
        self.width = width
        self.pad_x, self.pad_y = pad_x, pad_y  # only X, Y need pooling-divisibility padding (Z is never pooled)

        self.conv_layers = nn.ModuleList([
            SpectralConv4d(width, width, modes1, modes2, modes3, modes4) for _ in range(n_layers)
        ])
        self.linear_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(n_layers)])

        # Deliberately decoupled from `width` -- an earlier version derived
        # this as max(4, width // 4), which silently stayed at 4 for both
        # width=8 and width=16 (max(4,2)=4, max(4,4)=4), meaning a "bigger
        # model" test never actually gave the U-Net more capacity. Pass it
        # explicitly so this can be controlled and tested independently.
        if unet_base_channels is None:
            unet_base_channels = max(4, width // 4)
        self.unet = UNet3D_Lite(width, base_channels=unet_base_channels)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        self.fc1 = nn.Linear(width, width2)
        self.fc2 = nn.Linear(width2, out_dim)

    def forward(self, x):
        """
        x: (B, width, X, Y, Z, T).
        Runs the UNet3D_Lite branch once per timestep (folding T into the
        batch dim), then for each of `n_layers` layers combines the
        spectral conv output, a pointwise linear projection, and the
        (alpha-scaled) UNet contribution, GroupNorm-normalizes, and applies
        GELU. Crops off the pooling-divisibility padding on X/Y, then
        projects width -> out_dim channels via a 2-layer MLP.
        Returns: (B, X, Y, Z, T, out_dim).
        """
        B, C, X, Y, Z, T = x.shape

        x_unet = x.permute(0, 5, 1, 2, 3, 4).contiguous().view(B * T, C, X, Y, Z)
        unet_out = self.unet(x_unet).view(B, T, C, X, Y, Z).permute(0, 2, 3, 4, 5, 1).contiguous()

        for conv, lin, norm in zip(self.conv_layers, self.linear_layers, self.norm_layers):
            x1 = conv(x)
            x2 = lin(x.reshape(B, self.width, -1)).view(B, self.width, X, Y, Z, T)
            x = norm(x1 + x2 + self.alpha * unet_out)
            x = F.gelu(x)

        # remove the pooling-divisibility padding we added before this block
        if self.pad_x:
            x = x[:, :, :-self.pad_x, :, :, :]
        if self.pad_y:
            x = x[:, :, :, :-self.pad_y, :, :]

        x = x.permute(0, 2, 3, 4, 5, 1)  # -> (B, X, Y, Z, T, C)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# GPF (Global Parametric Fusion) -- FiLM-style conditioning (2026-07-27 fix)
# ---------------------------------------------------------------------------
class GlobalParametricFusion(nn.Module):
    """
    Predicts a per-channel (gamma, beta) pair applied to the spatial branch
    AFTER normalization, instead of being concatenated before a GroupNorm
    that would otherwise erase it (see module docstring for the full
    root-cause trace). `(1 + gamma)` rather than raw `gamma` so this starts
    as an identity transform (scale~1, shift~0) at initialization, since a
    freshly-initialized linear layer's output is small/near-zero -- avoids
    an arbitrary random rescaling of the spatial branch from the first
    forward pass.
    """
    def __init__(self, n_params, target_channels, hidden=64):
        """
        n_params: number of scalar parameters fed in (14 in this project).
        target_channels: number of channels in the spatial feature map this
        FiLM conditioning will be applied to.
        hidden: hidden layer width of the params -> (gamma, beta) MLP.
        """
        super().__init__()
        self.target_channels = target_channels
        self.mlp = nn.Sequential(nn.Linear(n_params, hidden), nn.ReLU(), nn.Linear(hidden, 2 * target_channels))

    def forward(self, params):
        """
        params: (B, n_params) raw scalar parameters.
        Returns: (gamma, beta), each (B, target_channels) -- the per-channel
        FiLM scale and shift to apply to the spatial branch.
        """
        out = self.mlp(params)  # (B, 2*C)
        gamma, beta = out.chunk(2, dim=-1)  # each (B, C)
        return gamma, beta


# ---------------------------------------------------------------------------
# Top-level single-field network: GPF + FNO/U-Net block, for ONE output field
# (T_K or p_MPa) -- two instances of this are used, matching PCNO's design
# ---------------------------------------------------------------------------
class FieldNet(nn.Module):
    """
    Single-field network: predicts one scalar field (T_K or p_MPa) over the
    full (X, Y, Z, T) grid from the static input channels and scalar
    params. Combines an input projection, FiLM conditioning via
    GlobalParametricFusion, and the repeated spectral+UNet block
    (Block4dWithUNet). Two instances of this class are used by
    ComsolFnoModel, one per output field.
    """
    def __init__(self, in_dim, n_params, grid_shape, width=8, width2=32,
                 modes1=4, modes2=4, modes3=4, modes4=3, unet_base_channels=None, n_layers=4):
        """
        in_dim: number of static input channels (9: phi, log10_perm,
                 x_norm, y_norm, z_norm, well_map, dist_inj, dist_prod,
                 plus the time-conditioning channel added by ComsolFnoModel)
        n_params: number of scalar params fed to GPF (14, after dropping
                   the constant well_radius)
        grid_shape: (X, Y, Z) spatial shape, e.g. (60, 30, 20)
        width/width2/modes*: architecture size -- start deliberately tiny
                              (matching PCNO's own "functional testing"
                              scale) given the 4GB GPU budget; scale up only
                              after confirming actual VRAM usage.
        unet_base_channels: U-Net capacity, independent of `width` (see
                              Block4dWithUNet docstring for why this needs
                              to be explicit rather than derived).
        n_layers: depth of the repeated spectral+UNet block (PCNO uses 4).
        """
        super().__init__()
        X, Y, Z = grid_shape
        self.pad_x = compute_pad_for_pooling(X)
        self.pad_y = compute_pad_for_pooling(Y)

        self.fc0 = nn.Linear(in_dim, width)
        self.global_fusion = GlobalParametricFusion(n_params, width)
        self.norm_fusion = nn.GroupNorm(8, width)  # normalizes the spatial branch ONLY now -- FiLM injects params after this

        self.block = Block4dWithUNet(width, width2, modes1, modes2, modes3, modes4,
                                      out_dim=1, pad_x=self.pad_x, pad_y=self.pad_y,
                                      unet_base_channels=unet_base_channels, n_layers=n_layers)

    def forward(self, x_static, params):
        """
        x_static: (B, in_dim, X, Y, Z, T) -- static channels broadcast across T
                  by the caller (see ComsolFnoModel.forward)
        params:   (B, n_params)
        Returns:  (B, X, Y, Z, T) -- single-field prediction
        """
        x = self.fc0(x_static.permute(0, 2, 3, 4, 5, 1))  # -> (B,X,Y,Z,T,width)
        x = x.permute(0, 5, 1, 2, 3, 4).contiguous()       # -> (B,width,X,Y,Z,T)

        # F.pad pads from the last dim backward: (T, Z, Y, X) order for this
        # 6D tensor -- pad list is [T_lo,T_hi, Z_lo,Z_hi, Y_lo,Y_hi, X_lo,X_hi].
        # Only X and Y need padding (for the U-Net's pooling divisibility);
        # T and Z are left untouched.
        if self.pad_x or self.pad_y:
            x = F.pad(x, [0, 0, 0, 0, 0, self.pad_y, 0, self.pad_x])

        x = self.norm_fusion(x)  # normalize the spatial branch BEFORE injecting params via FiLM
        gamma, beta = self.global_fusion(params)  # each (B, width)
        B, C = gamma.shape
        x = x * (1 + gamma.view(B, C, 1, 1, 1, 1)) + beta.view(B, C, 1, 1, 1, 1)

        out = self.block(x)  # (B, X, Y, Z, T, 1)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Well-output head (Option B): read the field prediction off at the well's
# grid cell, then upsample from n_field_time (9) to n_well_time (201) steps
# ---------------------------------------------------------------------------
class WellOutputHead(nn.Module):
    """
    Takes the T_K and p_MPa predictions at the injector and producer grid
    cells (4 scalar series x n_field_time each) PLUS the raw scalar params,
    and refines/upsamples them into the full n_well_time-step well output
    curves (T_prod_K, P_inj_MPa, P_prod_MPa -- dP_MPa is derived
    arithmetically afterward, not predicted, since it's exactly
    P_inj_MPa - P_prod_MPa, verified in Phase 1).

    `params` is included directly because well curves need 201 timesteps
    while the field prediction this head reads from only has n_field_time
    (9, from the step=5 grid) -- fast dynamics (e.g. early-time pressure
    transients, thermal breakthrough timing) plausibly happen faster than
    the 9-snapshot field grid can resolve at all, and are directly driven
    by params like injection rate/well spacing.
    """
    def __init__(self, n_field_time, n_well_time, n_params, hidden=64):
        """
        n_field_time: number of field snapshots (9) the well-cell readings
        are extracted from.
        n_well_time: number of well-output timesteps to produce (201).
        n_params: number of raw scalar params concatenated in alongside the
        field-derived series.
        hidden: hidden layer width of the upsampling MLP.
        """
        super().__init__()
        self.n_well_time = n_well_time
        # 4 field-derived series (T at inj, T at prod, P at inj, P at prod) + raw params -> 3 output series
        self.net = nn.Sequential(
            nn.Linear(n_field_time * 4 + n_params, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_well_time * 3),
        )

    def forward(self, t_at_inj, t_at_prod, p_at_inj, p_at_prod, params):
        """
        Concatenates the four field-derived well-cell series with the raw
        scalar params and passes them through the upsampling MLP to
        produce the full-resolution well-output curves.

        t_at_inj, t_at_prod, p_at_inj, p_at_prod: (B, n_field_time) --
        T_K/p_MPa field values read off at the injector/producer cell (see
        _read_at_well). params: (B, n_params).
        Returns: (B, n_well_time, 3) -- [T_prod_K, P_inj_MPa, P_prod_MPa].
        """
        B = t_at_inj.shape[0]
        x = torch.cat([t_at_inj, t_at_prod, p_at_inj, p_at_prod, params], dim=1)  # (B, n_field_time*4 + n_params)
        out = self.net(x).view(B, self.n_well_time, 3)
        return out  # channels: [T_prod_K, P_inj_MPa, P_prod_MPa]


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------
class ComsolFnoModel(nn.Module):
    """
    Top-level model for the COMSOL/FNO 3D geothermal surrogate: combines
    two FieldNet branches (4D FNO + FiLM-conditioned parameter fusion + 3D
    U-Net, one branch each for temperature and pressure) with a
    WellOutputHead that reads the field predictions off at the well
    locations and upsamples them into full-resolution well-output curves.

    Given one scenario's static spatial channels (geology + well-location
    fields, time-invariant), its scalar operational/rock-property
    parameters, and a well-location map, produces the full 4D temperature
    and pressure fields over the 9-snapshot uniform time grid, plus the
    three well-output curves (T_prod_K, P_inj_MPa, P_prod_MPa) over the
    full 201-timestep well horizon. See forward() for exact input/output
    shapes.
    """
    def __init__(self, in_dim=8, n_params=14, grid_shape=(60, 30, 20),
                 n_field_time=9, n_well_time=201,
                 width=8, width2=32, modes1=4, modes2=4, modes3=4, modes4=3,
                 unet_base_channels=None, n_layers=4):
        """
        Constructs the two FieldNet sub-networks (one each for T_K and
        p_MPa; each receives the static channels plus one appended
        time-conditioning channel) and the WellOutputHead, and precomputes
        the normalized snapshot-time values (0..1 over n_field_time) used
        to build that time-conditioning channel at forward time.

        in_dim: number of static input channels (8: phi, log10_perm,
        x_norm, y_norm, z_norm, well_map, dist_inj, dist_prod).
        n_params: number of scalar parameters fed to FiLM conditioning and
        the well-output head (14, after dropping the constant well_radius).
        grid_shape: (X, Y, Z) spatial shape, e.g. (60, 30, 20).
        n_field_time: number of field snapshots (9, from the step=5-year
        uniform time grid).
        n_well_time: number of well-output timesteps (201).
        width, width2, modes1, modes2, modes3, modes4, unet_base_channels,
        n_layers: architecture hyperparameters, passed through unchanged to
        each FieldNet.
        """
        super().__init__()
        self.grid_shape = grid_shape
        self.n_field_time = n_field_time

        # Time-conditioning fix (2026-07-27, carried over from the earlier model):
        # x_static used to be broadcast identically across every T snapshot
        # and GPF's params fusion was likewise T-invariant, so FieldNet's
        # input was exactly constant across T -- and SpectralConv4d (which
        # multiplies each FFT frequency bin independently, never mixing
        # bins) could only map a DC-only spectrum to another DC-only
        # spectrum, i.e. a time-constant output, regardless of training.
        # Fix: append one extra input channel carrying the normalized
        # snapshot time (0..1 over n_field_time), varying across T.
        self.register_buffer('time_values', torch.linspace(0.0, 1.0, n_field_time))

        self.t_net = FieldNet(in_dim + 1, n_params, grid_shape, width, width2, modes1, modes2, modes3, modes4,
                               unet_base_channels=unet_base_channels, n_layers=n_layers)
        self.p_net = FieldNet(in_dim + 1, n_params, grid_shape, width, width2, modes1, modes2, modes3, modes4,
                               unet_base_channels=unet_base_channels, n_layers=n_layers)
        self.well_head = WellOutputHead(n_field_time, n_well_time, n_params)

    def forward(self, x_static, params, well_map):
        """
        x_static: (B, in_dim, X, Y, Z) -- static channels (time-invariant)
        params:   (B, n_params)
        well_map: (B, X, Y, Z) -- +1 at injector cell, -1 at producer cell, 0 elsewhere
                  (used to read off predictions at the well locations)
        Returns dict with 'T_K' (B,X,Y,Z,T), 'p_MPa' (B,X,Y,Z,T), 'wells' (B, n_well_time, 3)
        """
        B = x_static.shape[0]
        T = self.n_field_time
        X, Y, Z = x_static.shape[-3:]
        x_static_bt = x_static.unsqueeze(-1).expand(-1, -1, -1, -1, -1, T)  # broadcast across time

        time_channel = self.time_values.view(1, 1, 1, 1, 1, T).expand(B, 1, X, Y, Z, T)
        x_with_time = torch.cat([x_static_bt, time_channel], dim=1)  # (B, in_dim+1, X, Y, Z, T)

        T_K = self.t_net(x_with_time, params)   # (B, X, Y, Z, T)
        p_MPa = self.p_net(x_with_time, params)  # (B, X, Y, Z, T)

        inj_mask = (well_map > 0.5)   # (B, X, Y, Z)
        prod_mask = (well_map < -0.5)

        t_at_inj = _read_at_well(T_K, inj_mask)     # (B, T)
        t_at_prod = _read_at_well(T_K, prod_mask)
        p_at_inj = _read_at_well(p_MPa, inj_mask)
        p_at_prod = _read_at_well(p_MPa, prod_mask)

        wells = self.well_head(t_at_inj, t_at_prod, p_at_inj, p_at_prod, params)

        return {'T_K': T_K, 'p_MPa': p_MPa, 'wells': wells}


def _read_at_well(field, well_mask_bool):
    """
    field: (B, X, Y, Z, T), well_mask_bool: (B, X, Y, Z) with exactly one
    True cell per batch element (injector or producer location -- a single
    middle-z-layer cell, per the well_map derivation in Phase 1's
    build_well_location_channels; NOT the well's whole vertical column,
    which would risk picking up invalid/placeholder cap-rock cells).
    Returns (B, T) -- the field's value at that cell, for every timestep.
    """
    B, X, Y, Z, T = field.shape
    counts = well_mask_bool.reshape(B, -1).sum(dim=1)
    assert torch.all(counts == 1), (
        f'_read_at_well expects exactly one marked cell per batch element, got counts={counts.tolist()}')
    mask = well_mask_bool.unsqueeze(-1).expand(-1, -1, -1, -1, T)  # (B,X,Y,Z,T)
    flat = field.reshape(B, -1, T)
    mask_flat = mask.reshape(B, -1, T)
    return (flat * mask_flat).sum(dim=1)


# ---------------------------------------------------------------------------
# Masked loss functions
# ---------------------------------------------------------------------------
def masked_mse(pred, target, mask):
    """mask: bool/float tensor, same shape as pred/target. Cells where
    mask is False contribute 0 to both the numerator and the denominator --
    excluded from the loss entirely, not zero-filled-and-averaged."""
    mask = mask.float()
    diff2 = (pred - target) ** 2 * mask
    return diff2.sum() / mask.sum().clamp(min=1.0)


def well_output_mse(pred_wells, target_wells):
    """pred_wells, target_wells: (B, n_well_time, 3) -- 100% finite, no masking needed."""
    return F.mse_loss(pred_wells, target_wells)


# ---------------------------------------------------------------------------
# Relative error metric -- matching pcno_lite/losses.py's established
# convention (mean(|pred-target|/|target|) * 100), which itself matches
# what the PCNO paper reports. The project's actual accuracy target is
# <=2% RE, so this is the metric that matters for judging real results --
# normalized MSE loss alone doesn't tell us how close we are to that.
#
# Must be computed on DENORMALIZED (real physical unit) values for the
# percentage to be physically meaningful -- callers are responsible for
# denormalizing pred/target using the Phase 1 stats before calling this.
# ---------------------------------------------------------------------------
def masked_relative_error(pred, target, mask, eps=1e-8):
    """Masked version of pcno_lite's relative_error: mean(|pred-target|/(|target|+eps)) * 100,
    computed only over cells where mask is True."""
    mask = mask.float()
    rel = torch.abs(pred - target) / (torch.abs(target) + eps)
    return (rel * mask).sum() / mask.sum().clamp(min=1.0) * 100.0


def relative_error(pred, target, eps=1e-8):
    """Unmasked version, for the well outputs (100% finite, no masking needed)."""
    return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean() * 100.0
