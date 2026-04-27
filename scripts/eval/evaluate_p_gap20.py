#!/usr/bin/env python
"""Evaluate the four phosphorus MACE/MACELES variants on the P-GAP-20 test set.

The script expects ASE-readable test structures with reference `energy` and
`forces` fields, such as the public phosphorus tests from libAtoms' testing
framework. It reports energy RMSE in meV/atom and force-component RMSE in
meV/Angstrom.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ase.io
import numpy as np
import torch

# MACE models saved by older training stacks may need full-object loading on
# newer PyTorch releases where `weights_only=True` can become the default.
_ORIGINAL_TORCH_LOAD = torch.load
torch.load = lambda *args, **kwargs: _ORIGINAL_TORCH_LOAD(  # type: ignore[assignment]
    *args, **{**kwargs, "weights_only": False}
)

from mace.calculators import MACECalculator  # noqa: E402

MODEL_FILES = {
    "MACE": "P_MACE_baseline_stagetwo.model",
    "MACE_GNB": "P_MACE_gnb_stagetwo.model",
    "MACELES": "P_MACELES_stagetwo.model",
    "MACELES_GNB": "P_MACELES_gnb_stagetwo.model",
}

GAP20_TABLE1_MEV = {
    "cryst_dist": (15, 186),
    "2D": (13, 129),
    "liq_12_03_01_liqP4": (8, 226),
    "liq_12_03_02_network": (11, 201),
    "liq_P4": (8, 226),
    "rss_rnd": (116, 694),
    "rss_200": (55, 382),
    "rss_3c": (58, 375),
}

DEFAULT_TEST_FILES = [
    "P_test_set.xyz",
    "Hittorf_ActaB1969_isolated_layer.xyz",
    "Hittorf_Angew2020_isolated_layer.xyz",
    "phosphorene_ribbon_armchair.xyz",
    "phosphorene_ribbon_single_layer.xyz",
    "phosphorene_ribbon_zigzag.xyz",
]

Record = Dict[str, object]
GroupKey = Tuple[str, str]


def rmse(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")


def read_reference(atoms):
    try:
        energy = atoms.get_potential_energy()
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        return None
    return energy, forces


def gather_model_errors(model_name: str, model_path: Path, test_files: List[Path], device: str):
    print(f"[{model_name}] loading {model_path}")
    calculator = MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype="float32",
    )
    grouped: Dict[GroupKey, List[Record]] = defaultdict(list)

    for test_file in test_files:
        frames = ase.io.read(str(test_file), index=":")
        print(f"[{model_name}] evaluating {test_file.name}: {len(frames)} frames")
        for frame_index, atoms in enumerate(frames):
            reference = read_reference(atoms)
            if reference is None:
                continue
            energy_ref, forces_ref = reference
            atoms.calc = calculator
            energy_pred = atoms.get_potential_energy()
            forces_pred = np.asarray(atoms.get_forces(), dtype=float)
            config_type = atoms.info.get("config_type", "unknown")
            grouped[(test_file.name, config_type)].append(
                {
                    "frame_index": frame_index,
                    "n_atoms": len(atoms),
                    "dE_atom": (energy_pred - energy_ref) / len(atoms),
                    "dF": (forces_pred - forces_ref).reshape(-1),
                }
            )
    return grouped


def summarise_group(records: List[Record], trim_top_frac: float):
    dE_atom = np.asarray([float(record["dE_atom"]) for record in records], dtype=float)
    dF_blocks = [np.asarray(record["dF"], dtype=float) for record in records]
    dF_all = np.concatenate(dF_blocks) if dF_blocks else np.asarray([], dtype=float)

    drop_count = int(np.floor(trim_top_frac * len(records)))
    keep = np.ones(len(records), dtype=bool)
    if drop_count > 0:
        worst = np.argsort(-np.abs(dE_atom))[:drop_count]
        keep[worst] = False

    dF_trim = np.concatenate([block for block, keep_block in zip(dF_blocks, keep) if keep_block])
    return {
        "n_frames": len(records),
        "E_full": rmse(dE_atom) * 1000,
        "F_full": rmse(dF_all) * 1000,
        "E_trim": rmse(dE_atom[keep]) * 1000,
        "F_trim": rmse(dF_trim) * 1000,
        "n_dropped": drop_count,
    }


def collect_rows(all_results, trim_top_frac: float):
    keys = sorted({key for model_results in all_results.values() for key in model_results})
    rows = []
    for file_name, config_type in keys:
        row = {"file": file_name, "config_type": config_type}
        for model_name, grouped in all_results.items():
            records = grouped.get((file_name, config_type), [])
            if not records:
                continue
            stats = summarise_group(records, trim_top_frac)
            row.setdefault("n_frames", stats["n_frames"])
            row[f"E_{model_name}"] = stats["E_full"]
            row[f"F_{model_name}"] = stats["F_full"]
            row[f"E_{model_name}_trim"] = stats["E_trim"]
            row[f"F_{model_name}_trim"] = stats["F_trim"]
            row[f"n_dropped_{model_name}"] = stats["n_dropped"]
        if file_name == "P_test_set.xyz" and config_type in GAP20_TABLE1_MEV:
            row["E_GAP20"], row["F_GAP20"] = GAP20_TABLE1_MEV[config_type]
        return_row = row
        rows.append(return_row)
    return rows


def write_csv(path: Path, rows, model_names: List[str], trimmed: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_trim" if trimmed else ""
    header = ["file", "config_type", "n_frames"]
    for model_name in model_names:
        header += [f"E_{model_name}", f"F_{model_name}"]
    header += ["E_GAP20", "F_GAP20"]

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            values = [row.get("file", ""), row.get("config_type", ""), row.get("n_frames", "")]
            for model_name in model_names:
                values += [
                    format_float(row.get(f"E_{model_name}{suffix}")),
                    format_float(row.get(f"F_{model_name}{suffix}")),
                ]
            values += [row.get("E_GAP20", ""), row.get("F_GAP20", "")]
            writer.writerow(values)


def format_float(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models",
        help="Directory containing the four saved .model files.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing P-GAP-20 test .xyz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
        help="Directory for generated benchmark CSV files.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trim-top-frac", type=float, default=0.02)
    parser.add_argument(
        "--test-files",
        nargs="*",
        default=DEFAULT_TEST_FILES,
        help="Test file names inside --test-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_files = [args.test_dir / name for name in args.test_files]
    missing_tests = [path for path in test_files if not path.exists()]
    if missing_tests:
        raise FileNotFoundError(f"Missing test files: {missing_tests}")

    all_results = {}
    for model_name, file_name in MODEL_FILES.items():
        model_path = args.models_dir / file_name
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        all_results[model_name] = gather_model_errors(model_name, model_path, test_files, args.device)

    rows = collect_rows(all_results, args.trim_top_frac)
    model_names = list(MODEL_FILES)
    write_csv(args.output_dir / "p_gap20_rmse_full_recomputed.csv", rows, model_names, trimmed=False)
    write_csv(args.output_dir / "p_gap20_rmse_trimmed_top2pct_recomputed.csv", rows, model_names, trimmed=True)
    print(f"Wrote benchmark CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
