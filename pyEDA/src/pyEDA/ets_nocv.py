import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from .utils import timer
from pyscf import scf, tools


class ETS_NOCV:
    """
    ETS-NOCV implementation for restricted SCF.

    Attributes
    ----------
    mf_molecule : scf.hf.SCF
        SCF object for the full adduct (supermolecule).
    mf_fragments : list[scf.hf.SCF]
        List of SCF objects for isolated fragments.
    """

    def __init__(self, mf_molecule: scf.hf.SCF, mf_fragments: list[scf.hf.SCF]):
        self.labels = ["restricted"]
        self.mf_molecule = mf_molecule
        self.mf_fragments = mf_fragments

        if not all(isinstance(frag, type(mf_molecule)) for frag in mf_fragments):
            raise TypeError(
                "mf_molecule and all mf_fragments must be of the same SCF type."
            )

        self.molecule_ovlp = self.mf_molecule.get_ovlp()
        self._initial_elec_energy = self._get_scf_elec_energy()

        self.C_ij_pro = None
        self.C_ij_frozen = None

        self.P_molecule = None
        self.P_fragments = None
        self.P_pro = None
        self.P_frozen = None
        self.dP_Pauli = None
        self.dP_orb = None

        self.EDA = {
            "Electrostatic": 0.0,
            "XC": 0.0,
            "Pauli": 0.0,
            "Orb": 0.0,
            "Dispersion": 0.0,
            "Interaction": 0.0,
            "Approximation Error": 0.0,
        }

        self.NOCV = {}
        self.densities = None

    def _get_scf_elec_energy(self) -> np.float64:
        summary = self.mf_molecule.scf_summary
        e1 = summary.get("e1")

        is_dft = hasattr(self.mf_molecule, "_numint")

        if is_dft:
            coul = summary.get("coul")
            exc = summary.get("exc")
            elec_energy = e1 + coul + exc
        else:
            e2 = summary.get("e2")
            elec_energy = e1 + e2

        return elec_energy

    @timer()
    def run(self) -> None:
        self.build_densities()

        self.EDA["Electrostatic"], self.EDA["XC"] = self.dE_elec_elec()
        self.EDA["Electrostatic"] += self.dE_nuc_nuc() + self.dE_elec_nuc()

        self.EDA["Pauli"], self.EDA["Orb"] = self.dE_FRZ()

        if all(
            "dispersion" in mol.scf_summary
            for mol in (self.mf_molecule, *self.mf_fragments)
        ):
            self.EDA["Dispersion"] = self.mf_molecule.scf_summary["dispersion"] - sum(
                mf_frag.scf_summary["dispersion"] for mf_frag in self.mf_fragments
            )

        self.EDA["Interaction"] = sum(
            self.EDA[key]
            for key in ["XC", "Electrostatic", "Pauli", "Orb", "Dispersion"]
        )

        self.scf_energy = self.mf_molecule.e_tot - sum(
            frag.e_tot for frag in self.mf_fragments
        )
        self.EDA["SCF"] = self.scf_energy
        self.EDA["E_Int - dE_SP"] = self.EDA["Interaction"] - self.scf_energy

        self.nocv()

    @timer()
    def build_densities(self) -> None:
        self.P_molecule = self.mf_molecule.make_rdm1()
        self.P_fragments = [mf.make_rdm1() for mf in self.mf_fragments]

        self.P_pro = np.array(self.P_fragments).sum(axis=0)
        self.P_frozen = self.build_frozen_density()

        self.dP_Pauli = self.P_frozen - self.P_pro
        self.dP_orb = self.P_molecule - self.P_frozen

    @timer()
    def build_frozen_density(self) -> NDArray[np.float64]:
        self.C_ij_pro = self.build_Cij_pro()
        self.C_ij_frozen = self.build_Cij_frozen(self.C_ij_pro)
        return 2.0 * self.C_ij_frozen @ self.C_ij_frozen.T

    @timer()
    def build_Cij_pro(self) -> NDArray[np.float64]:
        coeffs = [
            frag.mo_coeff[:, np.nonzero(frag.mo_occ)[0]] for frag in self.mf_fragments
        ]
        return np.hstack(coeffs)

    @timer()
    def build_Cij_frozen(self, C_ij_pro: NDArray[np.float64]) -> NDArray[np.float64]:
        S_pro = C_ij_pro.T @ self.molecule_ovlp @ C_ij_pro
        eigvals, eigvecs = np.linalg.eigh(S_pro)

        S_inv_sqrt = 1.0 / np.sqrt(eigvals)
        X = (eigvecs * S_inv_sqrt) @ eigvecs.T

        return C_ij_pro @ X

    @timer()
    def dE_elec_elec(self) -> tuple[np.float64, np.float64]:
        j_matrix = self.mf_molecule.get_j(dm=self.P_pro)
        j = 0.5 * np.einsum(
            "ij,ji->", self.P_tot(j_matrix), self.P_tot(self.P_pro), optimize=True
        )

        j_plus_xc = self.mf_molecule.energy_elec(
            dm=self.P_pro, h1e=self.molecule_ovlp * 0.0
        )[0]

        xc = j_plus_xc - j

        for mf_frag, P_frag in zip(self.mf_fragments, self.P_fragments):
            j_mat_f = mf_frag.get_j(dm=P_frag)
            j_f = 0.5 * np.einsum(
                "ij,ji->",
                self.P_tot(j_mat_f),
                self.P_tot(P_frag),
                optimize=True,
            )

            j -= j_f
            xc -= (
                self.mf_molecule.energy_elec(dm=P_frag, h1e=self.molecule_ovlp * 0.0)[0]
                - j_f
            )

        return j, xc

    @timer()
    def dE_nuc_nuc(self) -> np.float64:
        E_nuc_mol = self.mf_molecule.scf_summary["nuc"]
        E_nuc_frags = sum(mf.scf_summary["nuc"] for mf in self.mf_fragments)
        return E_nuc_mol - E_nuc_frags

    @timer()
    def dE_elec_nuc(self) -> np.float64:
        result = 0.0
        V_nuc_mol = self.mf_molecule.mol.intor("int1e_nuc")+ self.mf_molecule.mol.intor("ECPscalar")

        for mf_frag, P_frag in zip(self.mf_fragments, self.P_fragments):
            P_frag = self.P_tot(P_frag)
            V_nuc_frag = mf_frag.mol.intor("int1e_nuc") + mf_frag.mol.intor("ECPscalar")
            dV = V_nuc_mol - V_nuc_frag
            result += np.einsum("pq,pq->", dV, P_frag, optimize=True)

        return result

    def P_tot(self, P: NDArray[np.float64]) -> NDArray[np.float64]:
        return P

    @timer()
    def dE_FRZ(self) -> tuple[np.float64, np.float64]:
        E_FRZ = self.mf_molecule.energy_elec(dm=self.P_frozen)[0]
        dE_Pauli = E_FRZ - self.mf_molecule.energy_elec(dm=self.P_pro)[0]
        dE_orb = self._initial_elec_energy - E_FRZ
        return dE_Pauli, dE_orb

    @timer()
    def nocv(self) -> None:
        self.NOCV = {}

        S0 = self.molecule_ovlp
        sqrtS0 = linalg.sqrtm(S0)
        A0 = linalg.inv(sqrtS0)

        F_list, dP_list = self.get_FdP_lists()

        for label, dP, F in zip(self.labels, dP_list, F_list):
            self.solve_nocv(label, dP, F, sqrtS0, A0)

    @timer()
    def get_FdP_lists(self):
        return [self.build_F_ij_TS()], [self.dP_orb]

    @timer()
    def build_F_ij_TS(self) -> NDArray[np.float64]:
        h1 = self.mf_molecule.get_hcore()

        Veff_TS = self.mf_molecule.get_veff(dm=0.5 * (self.P_molecule + self.P_frozen))
        Veff_mol = self.mf_molecule.get_veff(dm=self.P_molecule)
        Veff_frozen = self.mf_molecule.get_veff(dm=self.P_frozen)

        F_ij_TS = (
            h1
            + (2.0 / 3.0) * Veff_TS
            + (1.0 / 6.0) * Veff_mol
            + (1.0 / 6.0) * Veff_frozen
        )

        return F_ij_TS

    @timer()
    def solve_nocv(self, label, dP, F, sqrtS0, A0) -> None:
        dP_orth = sqrtS0 @ dP @ sqrtS0
        dP_orth = 0.5 * (dP_orth + dP_orth.T)

        eigvals, eigvecs = np.linalg.eigh(dP_orth)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        coef_ao = A0 @ eigvecs

        F_orth = A0 @ F @ A0
        F_nocv = eigvecs.T @ F_orth @ eigvecs
        orb_ene = np.diag(F_nocv)

        self.NOCV[label] = {
            "eigenvalues": eigvals,
            "eigenvectors_orth": eigvecs,
            "coefficients_ao": coef_ao,
            "F_nocv": F_nocv,
            "orbital_energies": orb_ene,
        }

        self.NOCV[label]["total_orbital_energy"] = np.sum(eigvals * orb_ene)

    def write_molden(self) -> None:
        for label in self.NOCV:
            suffix = f"_{label}" if label != "restricted" else ""
            filename = f"NOCV{suffix}.molden"

            with open(filename, "w") as f:
                tools.molden.header(self.mf_molecule.mol, f)
                tools.molden.orbital_coeff(
                    self.mf_molecule.mol,
                    f,
                    self.NOCV[label]["coefficients_ao"],
                    ene=self.NOCV[label]["orbital_energies"],
                    occ=self.NOCV[label]["eigenvalues"],
                )

    def gen_densities(self, **density_kwargs):
        DENSITY_KWARGS = {"nx", "ny", "nz", "resolution", "margin"}
        filtered = {k: v for k, v in density_kwargs.items() if k in DENSITY_KWARGS}

        mol_density = tools.cubegen.density(
            mol=self.mf_molecule.mol,
            outfile="mol_density.cube",
            dm=self.P_tot(self.P_molecule),
            **filtered,
        )

        promol_density = tools.cubegen.density(
            mol=self.mf_molecule.mol,
            outfile="promol_density.cube",
            dm=self.P_tot(self.P_pro),
            **filtered,
        )

        frozen_density = tools.cubegen.density(
            mol=self.mf_molecule.mol,
            outfile="frozen_density.cube",
            dm=self.P_tot(self.P_frozen),
            **filtered,
        )

        self.densities = {
            "mol": mol_density,
            "pro": promol_density,
            "frozen": frozen_density,
            "kwargs": filtered,
        }

        return self.densities

    def gen_PauliOrb_density(self, **density_kwargs):
        if self.densities is None:
            self.gen_densities(**density_kwargs)

        pauli_dens = self.densities["frozen"] - self.densities["pro"]
        orb_dens = self.densities["mol"] - self.densities["frozen"]

        pauli_cube = tools.cubegen.Cube(self.mf_molecule.mol, **density_kwargs)
        pauli_cube.write(pauli_dens, "pauli.cube")

        orb_cube = tools.cubegen.Cube(self.mf_molecule.mol, **density_kwargs)
        orb_cube.write(orb_dens, "orb.cube")

    def gen_pair_density(self, pair_number: int, label="restricted", **kwargs):
        coef = self.NOCV[label]["coefficients_ao"]

        dm_1 = np.outer(coef[:, pair_number - 1], coef[:, pair_number - 1])
        dm_2 = np.outer(coef[:, -pair_number], coef[:, -pair_number])

        x = tools.cubegen.density(self.mf_molecule.mol, "x.cube", dm_1, **kwargs)
        y = tools.cubegen.density(self.mf_molecule.mol, "y.cube", dm_2, **kwargs)

        z = self.NOCV[label]["eigenvalues"][pair_number - 1] * (x - y)

        cb = tools.cubegen.Cube(self.mf_molecule.mol, **kwargs)
        cb.write(z, f"{pair_number}_pair_density_{label}.cube")


class U_ETS_NOCV(ETS_NOCV):
    """Unrestricted ETS-NOCV."""

    def __init__(self, mf_molecule, mf_fragments):
        super().__init__(mf_molecule, mf_fragments)
        self.labels = ["alpha", "beta"]

    @timer()
    def build_frozen_density(self) -> NDArray[np.float64]:
        self.C_ij_pro = self.build_Cij_pro()

        C_frz_a = self.build_Cij_frozen(self.C_ij_pro[0])
        C_frz_b = self.build_Cij_frozen(self.C_ij_pro[1])

        return np.array(
            [
                C_frz_a @ C_frz_a.T,
                C_frz_b @ C_frz_b.T,
            ]
        )

    @timer()
    def build_Cij_pro(self):
        coeffs_a = [
            frag.mo_coeff[0][:, np.nonzero(frag.mo_occ[0])[0]]
            for frag in self.mf_fragments
        ]
        coeffs_b = [
            frag.mo_coeff[1][:, np.nonzero(frag.mo_occ[1])[0]]
            for frag in self.mf_fragments
        ]
        return np.hstack(coeffs_a), np.hstack(coeffs_b)

    def P_tot(self, P):
        return P[0] + P[1]

    @timer()
    def get_FdP_lists(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dP = self.dP_orb
        F = self.build_F_ij_TS()
        return F, dP
