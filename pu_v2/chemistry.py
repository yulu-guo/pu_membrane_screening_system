from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


NCO_PATTERN = Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]")
OH_PATTERN = Chem.MolFromSmarts("[OX2H1]")
NH2_PATTERN = Chem.MolFromSmarts("[NX3;H2;!$(N-C=O);!$(N-C=N)]")
CARBOXYLIC_ACID_PATTERN = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
URETHANE_PATTERN = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2]")
UREA_PATTERN = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[NX3]")
DUMMY_PATTERN = Chem.MolFromSmarts("[#0]")


class ChemistryError(ValueError):
    """Raised when a monomer or formulation cannot be represented."""


def canonical_smiles(smiles: str, allow_polymer: bool = False) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ChemistryError("SMILES cannot be empty.")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ChemistryError("RDKit could not parse the supplied SMILES.")
    if not allow_polymer and any(
        atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()
    ):
        raise ChemistryError("Polymer attachment points are not allowed here.")
    return Chem.MolToSmiles(molecule, canonical=True)


def match_count(molecule: Chem.Mol, pattern: Chem.Mol) -> int:
    return len(molecule.GetSubstructMatches(pattern, uniquify=True))


def reactive_sites(molecule: Chem.Mol) -> dict[str, list[tuple[int, ...]]]:
    return {
        "isocyanate": list(
            molecule.GetSubstructMatches(NCO_PATTERN, uniquify=True)
        ),
        "hydroxyl": list(molecule.GetSubstructMatches(OH_PATTERN, uniquify=True)),
        "amine": list(molecule.GetSubstructMatches(NH2_PATTERN, uniquify=True)),
    }


def validate_monomer(
    smiles: str,
    role: str,
) -> tuple[str, Chem.Mol, dict[str, int]]:
    canonical = canonical_smiles(smiles, allow_polymer=False)
    molecule = Chem.MolFromSmiles(canonical)
    sites = reactive_sites(molecule)
    counts = {
        "isocyanate": len(sites["isocyanate"]),
        "hydroxyl": len(sites["hydroxyl"]),
        "amine": len(sites["amine"]),
    }

    if role == "isocyanate" and counts["isocyanate"] < 1:
        raise ChemistryError("An isocyanate monomer must contain N=C=O groups.")
    if role in {"polyol", "chain_extender"} and (
        counts["hydroxyl"] + counts["amine"] < 2
    ):
        raise ChemistryError(
            "A polyol or chain extender must contain at least two O-H or N-H2 sites."
        )
    if role == "crosslinker" and (
        counts["hydroxyl"] + counts["amine"] < 3
    ):
        raise ChemistryError("A crosslinker must contain at least three reactive sites.")
    return canonical, molecule, counts


def monomer_summary(
    smiles: str,
    role: str,
) -> dict[str, Any]:
    canonical, molecule, sites = validate_monomer(smiles, role)
    return {
        "canonical_smiles": canonical,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
        "molecular_weight": round(float(Descriptors.MolWt(molecule)), 4),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 4),
        "mol_logp": round(float(Descriptors.MolLogP(molecule)), 4),
        "bertz_ct": round(float(Descriptors.BertzCT(molecule)), 4),
        "h_bond_acceptors": int(rdMolDescriptors.CalcNumHBA(molecule)),
        "ring_count": int(molecule.GetRingInfo().NumRings()),
        "nco_sites": sites["isocyanate"],
        "oh_sites": sites["hydroxyl"],
        "nh2_sites": sites["amine"],
        "nco_site_count": int(sites["isocyanate"]),
        "oh_site_count": int(sites["hydroxyl"]),
        "nh2_site_count": int(sites["amine"]),
        "reactive_site_count": int(sum(sites.values())),
    }


def _remove_one_hydrogen(editable: Chem.RWMol, atom_index: int) -> None:
    atom = editable.GetAtomWithIdx(atom_index)
    explicit = atom.GetNumExplicitHs()
    if explicit > 0:
        atom.SetNumExplicitHs(explicit - 1)
    else:
        atom.SetNoImplicit(False)


def _form_urethane_or_urea(
    editable: Chem.RWMol,
    donor_atom: int,
    carbonyl_carbon: int,
    isocyanate_nitrogen: int,
) -> None:
    _remove_one_hydrogen(editable, donor_atom)
    editable.AddBond(
        donor_atom,
        carbonyl_carbon,
        Chem.BondType.SINGLE,
    )
    editable.RemoveBond(isocyanate_nitrogen, carbonyl_carbon)
    editable.AddBond(
        isocyanate_nitrogen,
        carbonyl_carbon,
        Chem.BondType.SINGLE,
    )


def _add_dummy_to_site(
    editable: Chem.RWMol,
    atom_index: int,
) -> None:
    _remove_one_hydrogen(editable, atom_index)
    dummy = editable.AddAtom(Chem.Atom(0))
    editable.AddBond(atom_index, dummy, Chem.BondType.SINGLE)


def _convert_nco_to_attachment(
    editable: Chem.RWMol,
    site: tuple[int, int, int],
) -> None:
    nitrogen, carbon, _oxygen = site
    editable.RemoveBond(nitrogen, carbon)
    editable.AddBond(nitrogen, carbon, Chem.BondType.SINGLE)
    _add_dummy_to_site(editable, carbon)


def _ordered_sequence(
    instances: list[dict[str, Any]],
    arrangement: str,
    custom_sequence: list[str] | None,
    random_seed: int,
) -> list[dict[str, Any]]:
    if arrangement == "custom" and custom_sequence:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            by_key.setdefault(instance["key"], []).append(instance)
        sequence = []
        for key in custom_sequence:
            if by_key.get(key):
                sequence.append(by_key[key].pop(0))
        for leftovers in by_key.values():
            sequence.extend(leftovers)
        return sequence

    if arrangement == "block":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            grouped.setdefault(instance["key"], []).append(instance)
        return [item for group in grouped.values() for item in group]

    if arrangement == "random":
        import random

        rng = random.Random(random_seed)
        shuffled = list(instances)
        rng.shuffle(shuffled)
        return shuffled

    # Alternating prioritizes an isocyanate followed by a soft/hydroxyl component.
    isocyanates = [
        item for item in instances if item["role"] == "isocyanate"
    ]
    partners = [
        item for item in instances if item["role"] != "isocyanate"
    ]
    sequence: list[dict[str, Any]] = []
    while isocyanates or partners:
        if isocyanates:
            sequence.append(isocyanates.pop(0))
        if partners:
            sequence.append(partners.pop(0))
    return sequence


@dataclass
class FormulationBuild:
    smiles: str
    sequence: list[str]
    connected_edges: int
    skipped_edges: int
    attachment_points: int
    component_fractions: dict[str, float]


def build_virtual_pu(
    selected_monomers: list[dict[str, Any]],
    arrangement: str = "alternating",
    custom_sequence: list[str] | None = None,
    random_seed: int = 42,
) -> FormulationBuild:
    """Construct a virtual PU repeat unit from arbitrarily ordered monomers."""

    if not selected_monomers:
        raise ChemistryError("At least one monomer is required.")
    if len(selected_monomers) > 24:
        raise ChemistryError("At most 24 monomer instances are supported per build.")

    instances: list[dict[str, Any]] = []
    combined: Chem.Mol | None = None
    offset = 0
    total_weight = 0.0
    role_mass: dict[str, float] = {}

    for index, selected in enumerate(selected_monomers):
        role = selected["role"]
        smiles = selected["smiles"]
        canonical, molecule, _sites = validate_monomer(smiles, role)
        key = selected.get("key") or selected.get("id") or f"m{index + 1}"
        weight = float(Descriptors.MolWt(molecule))
        total_weight += weight
        role_mass[role] = role_mass.get(role, 0.0) + weight

        if combined is None:
            combined = Chem.Mol(molecule)
        else:
            combined = Chem.CombineMols(combined, molecule)
        local_sites = reactive_sites(molecule)
        instances.append(
            {
                "key": key,
                "name": selected.get("name", key),
                "role": role,
                "canonical_smiles": canonical,
                "offset": offset,
                "sites": {
                    site_type: [
                        tuple(index + offset for index in site)
                        for site in site_list
                    ]
                    for site_type, site_list in local_sites.items()
                },
                "used": set(),
            }
        )
        offset += molecule.GetNumAtoms()

    sequence_instances = _ordered_sequence(
        instances,
        arrangement,
        custom_sequence,
        random_seed,
    )
    sequence_names = [item["key"] for item in sequence_instances]

    editable = Chem.RWMol(combined)
    connected_edges = 0
    skipped_edges = 0

    for left, right in zip(sequence_instances, sequence_instances[1:]):
        donor_instance = None
        nco_instance = None
        if left["role"] == "isocyanate" and right["role"] != "isocyanate":
            nco_instance, donor_instance = left, right
        elif right["role"] == "isocyanate" and left["role"] != "isocyanate":
            nco_instance, donor_instance = right, left

        if nco_instance is None or donor_instance is None:
            skipped_edges += 1
            continue

        nco_site = next(
            (
                site
                for site in nco_instance["sites"]["isocyanate"]
                if site not in nco_instance["used"]
            ),
            None,
        )
        donor_site = next(
            (
                ("hydroxyl", site)
                for site in donor_instance["sites"]["hydroxyl"]
                if site not in donor_instance["used"]
            ),
            None,
        )
        if donor_site is None:
            donor_site = next(
                (
                    ("amine", site)
                    for site in donor_instance["sites"]["amine"]
                    if site not in donor_instance["used"]
                ),
                None,
            )
        if nco_site is None or donor_site is None:
            skipped_edges += 1
            continue

        donor_type, donor_atoms = donor_site
        _form_urethane_or_urea(
            editable,
            donor_atoms[0],
            nco_site[1],
            nco_site[0],
        )
        nco_instance["used"].add(nco_site)
        donor_instance["used"].add(donor_atoms)
        if donor_type == "amine":
            donor_instance["used"].add(donor_atoms)
        connected_edges += 1

    for instance in instances:
        for site in instance["sites"]["hydroxyl"]:
            if site not in instance["used"]:
                _add_dummy_to_site(editable, site[0])
        for site in instance["sites"]["amine"]:
            if site not in instance["used"]:
                _add_dummy_to_site(editable, site[0])
        for site in instance["sites"]["isocyanate"]:
            if site not in instance["used"]:
                _convert_nco_to_attachment(editable, site)

    product = editable.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Chem.rdchem.MolSanitizeException as exc:
        raise ChemistryError(f"Could not sanitize the virtual PU structure: {exc}") from exc

    final_smiles = Chem.MolToSmiles(product, canonical=True)
    attachment_points = len(
        product.GetSubstructMatches(DUMMY_PATTERN, uniquify=True)
    )
    fractions = {
        role: mass / total_weight for role, mass in role_mass.items()
    }
    return FormulationBuild(
        smiles=final_smiles,
        sequence=sequence_names,
        connected_edges=connected_edges,
        skipped_edges=skipped_edges,
        attachment_points=attachment_points,
        component_fractions=fractions,
    )


def polymer_summary(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ChemistryError("Could not parse the generated polymer SMILES.")
    return {
        "molecular_weight": round(float(Descriptors.MolWt(molecule)), 4),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 4),
        "mol_logp": round(float(Descriptors.MolLogP(molecule)), 4),
        "molar_volume": round(float(Descriptors.MolMR(molecule)), 4),
        "bertz_ct": round(float(Descriptors.BertzCT(molecule)), 4),
        "h_bond_acceptors": int(rdMolDescriptors.CalcNumHBA(molecule)),
        "ring_count": int(molecule.GetRingInfo().NumRings()),
        "vsa_estate8": round(float(Descriptors.VSA_EState8(molecule)), 4),
        "peoe_vsa8": round(float(Descriptors.PEOE_VSA8(molecule)), 4),
        "nhco_groups": match_count(molecule, URETHANE_PATTERN)
        + match_count(molecule, UREA_PATTERN),
        "urethane_groups": match_count(molecule, URETHANE_PATTERN),
        "urea_groups": match_count(molecule, UREA_PATTERN),
        "attachment_points": match_count(molecule, DUMMY_PATTERN),
    }


@dataclass(frozen=True)
class MonomerCatalog:
    frame: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path) -> "MonomerCatalog":
        if not path.exists():
            raise FileNotFoundError(f"Monomer catalog not found: {path}")
        frame = pd.read_csv(path)
        required = {"id", "name", "role", "smiles"}
        missing = required.difference(frame.columns)
        if missing:
            raise ChemistryError(
                f"Monomer catalog is missing columns: {sorted(missing)}"
            )
        return cls(frame=frame)

    def get(self, monomer_id: str) -> dict[str, Any]:
        row = self.frame.loc[self.frame["id"] == monomer_id]
        if row.empty:
            raise ChemistryError(f"Unknown monomer id: {monomer_id}")
        return row.iloc[0].to_dict()

    def by_role(self, role: str) -> pd.DataFrame:
        return self.frame.loc[self.frame["role"] == role].copy()

    def as_payload(self) -> list[dict[str, Any]]:
        return self.frame.to_dict(orient="records")
