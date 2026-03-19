from functools import wraps
import time
import warnings

from pyscf import gto


def index_parse(spec: str) -> list[int]:
    """
    Parse atom index specification like "1-3,5,7" into a sorted list of indices.
    Indices are assumed to be zero-based (PySCF convention).
    """
    if not spec.strip():
        raise ValueError("Empty fragment specification")

    indices: set[int] = set()
    parts = spec.replace(" ", "").split(",")

    for part in parts:
        if not part:
            continue

        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                if start > end:
                    raise ValueError(f"Incorrect range {part}")
                indices.update(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Range error '{part}': {e}")
        else:
            try:
                indices.add(int(part))
            except ValueError as e:
                raise ValueError(f"Incorrect atom index '{part}': {e}")

    return sorted(indices)


def make_frag_geom(
    molecule: gto.Mole,
    fragment_atoms,
):
    """
    Build fragment geometry using ghost atoms for excluded atoms.

    Parameters
    ----------
    molecule : gto.Mole
        Reference full molecule.
    fragment_atoms : str or list[int]
        Atom indices belonging to the fragment.

    Returns
    -------
    list[list]
        Geometry suitable for gto.M(atom=...).
    """
    if isinstance(fragment_atoms, str):
        fragment_atoms = index_parse(fragment_atoms)

    fragment_structure = [list(atom) for atom in molecule._atom]

    for idx, atom in enumerate(fragment_structure):
        if idx not in fragment_atoms:
            atom[0] = "GHOST-" + atom[0]

    return fragment_structure


def check_structure(molecule: gto.Mole, fragment_list: list[gto.Mole]) -> bool:
    """
    Check consistency between full molecule and a list of fragment molecules.

    Verifies:
    - atomic structure consistency (ignoring ghost atoms)
    - total charge conservation
    - total electron count conservation
    - alpha/beta electron consistency (warning only)
    """
    structure = [
        atom
        for mol in fragment_list
        for atom in mol._atom
        if not (isinstance(atom[0], str) and atom[0].lower().startswith(("ghost", "x")))
    ]

    adduct = gto.Mole(atom=structure)
    adduct.spin = sum(frag.spin for frag in fragment_list)
    adduct.charge = sum(frag.charge for frag in fragment_list)
    adduct.build(unit="BOHR")

    if not gto.same_mol(molecule, adduct, cmp_basis=False):
        raise ValueError(
            "Full molecule and composition of fragments are not consistent"
        )

    if adduct.charge != molecule.charge:
        raise ValueError(
            f"Molecule charge ({molecule.charge}) != "
            f"sum of fragment charges ({adduct.charge})"
        )

    total_alpha = sum(frag.nelec[0] for frag in fragment_list)
    total_beta = sum(frag.nelec[1] for frag in fragment_list)

    mol_alpha, mol_beta = molecule.nelec

    total_frag = total_alpha + total_beta
    total_mol = mol_alpha + mol_beta

    if total_frag != total_mol:
        raise ValueError(
            f"Electron count mismatch: molecule [{total_mol}] "
            f"vs fragments [{total_frag}]"
        )

    if total_alpha != mol_alpha or total_beta != mol_beta:
        warnings.warn(
            f"Alpha/Beta electron mismatch: molecule [{mol_alpha}, {mol_beta}] "
            f"vs fragments [{total_alpha}, {total_beta}]. "
            f"Check fragment electronic configurations."
        )

    return True


def timer():
    """
    Simple timing decorator collecting execution times on the wrapped function.
    """

    def decorator(func):
        func.execution_times = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            duration = end - start
            print(f"{func.__name__} took {duration:.4f} seconds")
            func.execution_times.append(duration)

            return result

        return wrapper

    return decorator
