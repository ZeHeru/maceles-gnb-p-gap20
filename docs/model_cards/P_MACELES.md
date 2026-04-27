# P_MACELES

## Intended use

`P_MACELES_stagetwo.model` adds the learned LES long-range channel to the short-range MACE backbone. Use it for ablation studies that isolate the effect of learned long-range interactions without the analytical GNB dispersion term.

## Training setup

- Architecture: `MACELES`
- Long-range terms: learned LES only
- Training file: `P_GAP_20_fitting_data.xyz`
- Labels: `energy` and `forces`
- Validation split: 2%, seed 123
- Short-range cutoff: 4.5 Angstrom
- SWA stage: enabled from epoch 400

## Strengths

- Useful controlled comparison against `P_MACE_baseline` and `P_MACELES_gnb`.
- Often improves high-pressure or long-range-sensitive behavior compared with a purely short-range model.

## Limitations

- Does not include the analytical GNB dispersion baseline.
- For layered or exfoliation-like systems, compare against the GNB variants before using this as the production model.
- Extrapolation to chemically different systems, impurities, adsorbates, or non-phosphorus species is unsupported.
