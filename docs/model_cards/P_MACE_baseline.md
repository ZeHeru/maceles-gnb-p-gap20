# P_MACE_baseline

## Intended use

`P_MACE_baseline_stagetwo.model` is the short-range-only MACE baseline for elemental phosphorus. Use it when you need a simple ablation baseline or when the target structures are dominated by local covalent P environments.

## Training setup

- Architecture: `MACE`
- Long-range terms: none
- Training file: `P_GAP_20_fitting_data.xyz`
- Labels: `energy` and `forces`
- Validation split: 2%, seed 123
- Short-range cutoff: 4.5 Angstrom
- SWA stage: enabled from epoch 400

## Strengths

- Strong and stable baseline for bulk crystalline and many local phosphorus environments.
- Useful reference for isolating the effect of LES and GNB terms.
- Does not depend on additional post-hoc dispersion calculators.

## Limitations

- Does not include an explicit long-range dispersion or electrostatic term.
- Not recommended as the primary model for exfoliation, interlayer binding, or long-range layered phosphorus simulations.
- Extrapolation to chemically different systems, impurities, adsorbates, or non-phosphorus species is unsupported.
