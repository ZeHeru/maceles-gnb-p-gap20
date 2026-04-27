# P_MACELES_gnb

## Intended use

`P_MACELES_gnb_stagetwo.model` is the most physically complete model in this repository: short-range MACE, learned LES long-range interactions, and the GNB analytical dispersion term. It is the recommended starting point for ordered crystals, 2D/phosphorene-like structures, molecular liquids, and network liquids of elemental phosphorus.

## Training setup

- Architecture: `MACELES`
- Long-range terms: learned LES plus GNB analytical dispersion
- GNB cutoff: 12.0 Angstrom
- Training file: `P_GAP_20_fitting_data.xyz`
- Labels: `energy` and `forces`
- Validation split: 2%, seed 123
- Short-range cutoff: 4.5 Angstrom
- SWA stage: enabled from epoch 400

## Strengths

- Best default choice when long-range physics matters.
- Strong benchmark performance on crystalline, 2D, liquid, and high-pressure crystalline phosphorus subsets.
- Closest in spirit to a short-range ML model augmented by physically motivated long-range corrections.

## Limitations

- Full RMSE on very random high-energy RSS initial structures can be dominated by rare outlier frames.
- Avoid using this model blindly for unconstrained random structure search seeds without geometry sanity checks.
- Extrapolation to chemically different systems, impurities, adsorbates, or non-phosphorus species is unsupported.
