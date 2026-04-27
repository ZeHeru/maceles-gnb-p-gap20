###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Optional
import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class ChebychevBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.register_buffer(
            "n",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()).unsqueeze(
                0
            ),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x.repeat(1, self.num_basis)
        n = self.n.repeat(len(x), 1)
        return torch.special.chebyshev_polynomial_t(x, n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis},"
        )


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


@compile_mode("script")
class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(self, p=6, trainable=False, **kwargs):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated. r_max is determined from the covalent radii."
            )

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={self.c})"


@compile_mode("script")
class GNBBasis(torch.nn.Module):
    """GNB (General Nonbonded) long-range dispersion augmentation
    based on Luo & Goddard, J. Chem. Theory Comput. 2025, 21, 499-515.

    Original GNB:  E_GNB = E_PR + E_LD
      E_PR = exp(-(r - β_ij) / s_ij)          # Pauli repulsion (dropped here)
      E_LD = -C6_ij / (R_ij^6 + r^6)          # London dispersion

    This implementation keeps ONLY E_LD and damps it with a Becke-Johnson
    (D3(BJ)) style switch so the term vanishes in the covalent / short-range
    region already captured by the MACE short-range NN and the D3 correction
    embedded in the ωB97M-D3(BJ) training labels.

    Damping:  f_damp(r) = 1 / (1 + 6 * (r / r0_ij)^(-alpha)),  alpha = 14
              r0_ij     = a1 * sqrt(R_ij) + a2          (BJ radius)
    Defaults a1=0.4, a2=4.0 Å, alpha=14 (standard D3(BJ) form).

    Combination rules (unchanged from GNB paper):
    - C6_ij = sqrt(C6_i * C6_j)  (geometric mean)
    - R_ij  = sqrt(R_i * R_j)    (geometric mean)
    """

    p: torch.Tensor

    def __init__(
        self,
        cutoff_lr: Optional[float] = None,
        p: int = 6,
        bj_a1: float = 0.4,
        bj_a2: float = 4.0,
        bj_alpha: float = 14.0,
        **kwargs,
    ):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated for GNBBasis. Cutoff is handled by the envelope function."
            )

        # Short-range BJ damping parameters
        self.register_buffer(
            "bj_a1", torch.tensor(bj_a1, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "bj_a2", torch.tensor(bj_a2, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "bj_alpha", torch.tensor(bj_alpha, dtype=torch.get_default_dtype())
        )

        if cutoff_lr is not None:
          self.register_buffer(
              "cutoff_lr", torch.tensor(cutoff_lr, dtype=torch.get_default_dtype())
          )
          self.register_buffer(
              "cutoff_start", torch.tensor(cutoff_lr - 2.0, dtype=torch.get_default_dtype())
          )
          self.register_buffer("p", torch.tensor(p, dtype=torch.int))
          self.register_buffer("r_max_scaled", torch.tensor(1.0, dtype=torch.get_default_dtype()))
          self.use_cutoff = True
        else:
            self.use_cutoff = False
        # GNB parameters for elements H-Rn (atomic numbers 1-86, excluding lanthanides)
        # Format: [s, beta, R, C6] for each element
        # Index 0 is unused (no element with Z=0)
        gnb_params = torch.zeros((87, 4), dtype=torch.get_default_dtype())

        # Period 1
        gnb_params[1] = torch.tensor([0.2772, 1.7504, 3.6516, 95.99])  # H
        gnb_params[2] = torch.tensor([0.2425, 2.0870, 2.1843, 40.67])  # He

        # Period 2
        gnb_params[3] = torch.tensor([0.3266, 2.3073, 1.2711, 70.21])  # Li
        gnb_params[4] = torch.tensor([0.3964, 2.8811, 3.3497, 114.51])  # Be
        gnb_params[5] = torch.tensor([0.3121, 2.6366, 2.7079, 152.36])  # B
        gnb_params[6] = torch.tensor([0.2455, 3.0189, 1.8219, 184.28])  # C (default sp3)
        gnb_params[7] = torch.tensor([0.2743, 2.9432, 2.4667, 482.54])  # N (default CN=3)
        gnb_params[8] = torch.tensor([0.2577, 2.7407, 2.3650, 405.57])  # O
        gnb_params[9] = torch.tensor([0.2791, 2.5863, 1.5062, 218.45])  # F
        gnb_params[10] = torch.tensor([0.2378, 2.5307, 1.8233, 174.81])  # Ne

        # Alternative parameters for C and N based on hybridization/coordination
        # These will be selected dynamically in forward()
        self.register_buffer(
            "C_sp2_params",
            torch.tensor([0.2808, 3.0548, 2.2348, 429.69], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "C_sp3_params",
            torch.tensor([0.2455, 3.0189, 1.8219, 184.28], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "N_CN2_params",
            torch.tensor([0.2996, 3.0200, 2.6454, 720.18], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "N_CN3_params",
            torch.tensor([0.2743, 2.9432, 2.4667, 482.54], dtype=torch.get_default_dtype())
        )

        # Period 3
        gnb_params[11] = torch.tensor([0.2592, 2.5679, 1.3974, 181.70])  # Na
        gnb_params[12] = torch.tensor([0.3837, 2.9613, 3.3515, 263.02])  # Mg
        gnb_params[13] = torch.tensor([0.3013, 3.2377, 3.0102, 228.10])  # Al
        gnb_params[14] = torch.tensor([0.3202, 3.2749, 3.1629, 359.43])  # Si
        gnb_params[15] = torch.tensor([0.3399, 3.7491, 3.2554, 3222.12])  # P
        gnb_params[16] = torch.tensor([0.3215, 3.6718, 2.9539, 2144.49])  # S
        gnb_params[17] = torch.tensor([0.3071, 3.4180, 3.0368, 2072.46])  # Cl
        gnb_params[18] = torch.tensor([0.2752, 3.3306, 2.6598, 1357.42])  # Ar

        # Period 4
        gnb_params[19] = torch.tensor([0.2264, 2.9736, 4.0877, 1406.65])  # K
        gnb_params[20] = torch.tensor([0.4015, 3.4464, 4.1275, 1058.36])  # Ca
        gnb_params[21] = torch.tensor([0.9388, 4.5955, 9.7282, 11498.73])  # Sc
        gnb_params[22] = torch.tensor([0.6048, 3.5334, 8.5322, 3361.33])  # Ti
        gnb_params[23] = torch.tensor([0.3678, 3.0680, 7.2344, 2095.91])  # V
        gnb_params[24] = torch.tensor([0.2848, 2.8393, 5.3605, 1049.31])  # Cr
        gnb_params[25] = torch.tensor([0.3550, 3.0550, 3.7180, 966.27])  # Mn
        gnb_params[26] = torch.tensor([0.4406, 3.2628, 3.6408, 1571.36])  # Fe
        gnb_params[27] = torch.tensor([0.4124, 3.2622, 3.4961, 1183.59])  # Co
        gnb_params[28] = torch.tensor([0.3139, 2.7490, 3.5108, 787.76])  # Ni
        gnb_params[29] = torch.tensor([0.3345, 2.9097, 3.0537, 563.93])  # Cu
        gnb_params[30] = torch.tensor([0.3391, 2.9024, 3.0261, 592.91])  # Zn
        gnb_params[31] = torch.tensor([0.3146, 3.3924, 3.1735, 430.82])  # Ga
        gnb_params[32] = torch.tensor([0.3327, 3.2804, 3.1773, 812.57])  # Ge
        gnb_params[33] = torch.tensor([0.4040, 3.8757, 3.8357, 4533.53])  # As
        gnb_params[34] = torch.tensor([0.3343, 3.8562, 3.1109, 3440.92])  # Se
        gnb_params[35] = torch.tensor([0.3242, 3.7067, 3.2122, 3859.82])  # Br
        gnb_params[36] = torch.tensor([0.2920, 3.6457, 2.8263, 2729.60])  # Kr

        # Period 5
        gnb_params[37] = torch.tensor([0.2677, 3.4421, 2.4120, 1864.19])  # Rb
        gnb_params[38] = torch.tensor([0.3446, 3.5590, 1.8940, 1175.73])  # Sr
        gnb_params[39] = torch.tensor([1.0038, 5.7293, 11.2061, 32141.18])  # Y
        gnb_params[40] = torch.tensor([0.6146, 5.0750, 6.8210, 27655.14])  # Zr
        gnb_params[41] = torch.tensor([0.4244, 3.7156, 7.2367, 2864.20])  # Nb
        gnb_params[42] = torch.tensor([0.4882, 3.6853, 3.9010, 3563.45])  # Mo
        gnb_params[43] = torch.tensor([0.4439, 3.6987, 4.0857, 3266.43])  # Tc
        gnb_params[44] = torch.tensor([0.4736, 3.7826, 4.0450, 3967.23])  # Ru
        gnb_params[45] = torch.tensor([0.4039, 3.4706, 3.4813, 2233.82])  # Rh
        gnb_params[46] = torch.tensor([0.3188, 3.3433, 3.0487, 1393.49])  # Pd
        gnb_params[47] = torch.tensor([0.2948, 3.1186, 2.7795, 1315.09])  # Ag
        gnb_params[48] = torch.tensor([0.3012, 3.0853, 2.8673, 1311.47])  # Cd
        gnb_params[49] = torch.tensor([0.3351, 3.5452, 3.3339, 1460.56])  # In
        gnb_params[50] = torch.tensor([0.3134, 3.2997, 3.0086, 1662.99])  # Sn
        gnb_params[51] = torch.tensor([0.4081, 4.1277, 3.9919, 8089.97])  # Sb
        gnb_params[52] = torch.tensor([0.3585, 4.1795, 3.4209, 6887.05])  # Te
        gnb_params[53] = torch.tensor([0.3574, 4.0988, 3.5649, 8799.32])  # I
        gnb_params[54] = torch.tensor([0.3134, 4.0387, 3.0288, 6136.50])  # Xe

        # Period 6
        gnb_params[55] = torch.tensor([0.2872, 3.8468, 2.2620, 3757.31])  # Cs
        gnb_params[56] = torch.tensor([0.3788, 4.0368, 1.3837, 2561.18])  # Ba
        gnb_params[57] = torch.tensor([1.0851, 6.1365, 12.1710, 66580.83])  # La
        # Skip lanthanides (58-71)
        gnb_params[72] = torch.tensor([0.5577, 5.2130, 6.0791, 27593.76])  # Hf
        gnb_params[73] = torch.tensor([0.5308, 4.7637, 5.7661, 15364.65])  # Ta
        gnb_params[74] = torch.tensor([0.3926, 3.6018, 3.6366, 2734.50])  # W
        gnb_params[75] = torch.tensor([0.5351, 3.9552, 4.2410, 4801.82])  # Re
        gnb_params[76] = torch.tensor([0.4701, 4.0278, 4.1348, 5685.94])  # Os
        gnb_params[77] = torch.tensor([0.3661, 3.6123, 3.4213, 2786.00])  # Ir
        gnb_params[78] = torch.tensor([0.3411, 3.6066, 3.2486, 2699.79])  # Pt
        gnb_params[79] = torch.tensor([0.3164, 3.3822, 2.9588, 2282.60])  # Au
        gnb_params[80] = torch.tensor([0.3047, 3.3750, 2.9381, 2476.79])  # Hg
        gnb_params[81] = torch.tensor([0.2802, 3.3964, 2.7711, 2988.70])  # Tl
        gnb_params[82] = torch.tensor([0.2682, 3.2481, 2.5816, 2506.63])  # Pb
        gnb_params[83] = torch.tensor([0.3876, 4.1225, 3.7850, 8916.84])  # Bi
        gnb_params[84] = torch.tensor([0.3679, 4.2313, 3.5381, 8694.22])  # Po
        gnb_params[85] = torch.tensor([0.3680, 4.2455, 3.6985, 11821.61])  # At
        gnb_params[86] = torch.tensor([0.3195, 4.1784, 3.0551, 8410.64])  # Rn

        self.register_buffer("gnb_params", gnb_params)
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def _compute_coordination_numbers(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute coordination number for each node based on edge connectivity."""
        # Count neighbors for each node
        receiver = edge_index[1]
        coord_numbers = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        coord_numbers.scatter_add_(0, receiver, torch.ones_like(receiver))
        return coord_numbers

    def _get_adaptive_params(
        self,
        Z: torch.Tensor,
        coord_numbers: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get GNB parameters with adaptive selection for C and N based on coordination."""
        params = self.gnb_params[Z.squeeze(-1)]  # [n_edges, 4]

        # For carbon (Z=6): use sp2 params if CN <= 3, sp3 params if CN = 4
        is_carbon = (Z.squeeze(-1) == 6)
        if is_carbon.any():
            carbon_cn = coord_numbers[node_indices[is_carbon]]
            is_sp2_carbon = is_carbon.clone()
            is_sp2_carbon[is_carbon] = (carbon_cn <= 3)
            is_sp3_carbon = is_carbon.clone()
            is_sp3_carbon[is_carbon] = (carbon_cn == 4)

            if is_sp2_carbon.any():
                params[is_sp2_carbon] = self.C_sp2_params
            if is_sp3_carbon.any():
                params[is_sp3_carbon] = self.C_sp3_params

        # For nitrogen (Z=7): use CN=2 params if CN <= 2, CN=3 params if CN >= 3
        is_nitrogen = (Z.squeeze(-1) == 7)
        if is_nitrogen.any():
            nitrogen_cn = coord_numbers[node_indices[is_nitrogen]]
            is_n_cn2 = is_nitrogen.clone()
            is_n_cn2[is_nitrogen] = (nitrogen_cn <= 2)
            is_n_cn3 = is_nitrogen.clone()
            is_n_cn3[is_nitrogen] = (nitrogen_cn >= 3)

            if is_n_cn2.any():
                params[is_n_cn2] = self.N_CN2_params
            if is_n_cn3.any():
                params[is_n_cn3] = self.N_CN3_params

        return params

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)

        # Compute coordination numbers for adaptive parameter selection
        num_nodes = node_attrs.size(0)
        coord_numbers = self._compute_coordination_numbers(edge_index, num_nodes)

        # Get GNB parameters with adaptive selection for C and N
        params_u = self._get_adaptive_params(Z_u, coord_numbers, sender)
        params_v = self._get_adaptive_params(Z_v, coord_numbers, receiver)

        # Apply combination rules
        s_ij = (params_u[:, 0] + params_v[:, 0]) / 2.0  # arithmetic mean
        beta_ij = (params_u[:, 1] + params_v[:, 1]) / 2.0  # arithmetic mean
        R_ij = torch.sqrt(params_u[:, 2] * params_v[:, 2])  # geometric mean
        C6_ij = torch.sqrt(params_u[:, 3] * params_v[:, 3])  # geometric mean

        r = x.squeeze(-1)

        # London dispersion only. Pauli repulsion is dropped: it is already
        # captured by the MACE short-range NN, and GNB's exp form explodes on
        # covalent bonds (hundreds of kcal/mol at r ≈ 1.5 Å), which would force
        # the NN to learn a massive cancellation and destabilise training.
        # E_LD = -C6_ij / (R_ij^6 + r^6)
        R6 = torch.pow(R_ij, 6)
        r6 = torch.pow(r, 6)
        E_LD = -C6_ij / (R6 + r6)

        # Becke-Johnson short-range damping (D3(BJ) style).
        # f_damp(r) = 1 / (1 + 6 * (r / r0_ij)^(-alpha))
        # With alpha = 14, f_damp ≈ 0 for r < r0_ij and ≈ 1 for r > r0_ij.
        # r0_ij is a pair-specific onset radius built from the GNB dispersion
        # length scale R_ij (~vdW radius proxy).
        r0_ij = self.bj_a1 * torch.sqrt(R_ij) + self.bj_a2
        # Guard against r = 0; edges always have r > 0 in practice, but add a
        # small epsilon to keep the power stable for autograd.
        r_safe = torch.clamp(r, min=1e-6)
        f_damp = 1.0 / (1.0 + 6.0 * torch.pow(r_safe / r0_ij, -self.bj_alpha))

        # Total per-edge contribution (kcal/mol -> eV: × 0.043364)
        # Factor 0.5 because each i<j pair appears in both directions in edge_index.
        v_edges = 0.5 * (E_LD * f_damp).unsqueeze(-1) * 0.043364

        if self.use_cutoff:
            r_scaled = (x.squeeze(-1) - self.cutoff_start) / (self.cutoff_lr - self.cutoff_start)
            r_scaled = torch.clamp(r_scaled, 0.0, 1.0) 

            envelope = PolynomialCutoff.calculate_envelope(
                r_scaled.unsqueeze(-1),
                self.r_max_scaled,
                self.p
            ).squeeze(-1)

            v_edges = v_edges * envelope.unsqueeze(-1)

        # Sum over edges to get per-node energies
        V_GNB = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_GNB.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


@compile_mode("script")
class AgnesiTransform(torch.nn.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """
    Tanh-based smooth transformation:
        T(x) = p1 + (x - p1)*0.5*[1 + tanh(alpha*(x - m))],
    which smoothly transitions from ~p1 for x << p1 to ~x for x >> r0.
    """

    def __init__(self, alpha: float = 4.0, trainable=False):
        """
        Args:
            p1 (float): Lower "clamp" point.
            alpha (float): Steepness; if None, defaults to ~6/(r0-p1).
            trainable (bool): Whether to make parameters trainable.
        """
        super().__init__()
        # Initialize parameters
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_0 based on atomic information.

        Args:
            node_attrs (torch.Tensor): Node attributes (one-hot encoding of atomic numbers).
            edge_index (torch.Tensor): Edge index indicating connections.
            atomic_numbers (torch.Tensor): Atomic numbers.

        Returns:
            torch.Tensor: r_0 values for each edge.
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        return r_0

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:

        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers)
        p_0 = (3 / 4) * r_0
        p_1 = (4 / 3) * r_0
        m = 0.5 * (p_0 + p_1)
        alpha = self.alpha / (p_1 - p_0)
        s_x = 0.5 * (1.0 + torch.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha.item():.4f})"


class RadialMLP(torch.nn.Module):
    """
    Construct a radial MLP (Linear → LayerNorm → SiLU) stack
    given a list of channel sizes, following ESEN / FairChem.
    """

    def __init__(self, channels_list) -> None:
        super().__init__()

        modules = []
        in_channels = channels_list[0]

        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if idx < len(channels_list) - 1:
                modules.append(torch.nn.LayerNorm(out_channels))
                modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)
        self.hs = channels_list

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
