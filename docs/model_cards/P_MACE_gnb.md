# P_MACE_gnb

## Intended use

`P_MACE_gnb_stagetwo.model` combines the short-range MACE model with the GNB analytical dispersion term. Use it when a physically motivated dispersion correction is desired but the LES channel should be excluded for ablation or robustness testing.

## Training setup

- Architecture: `MACE`
- Long-range terms: GNB analytical dispersion
- GNB cutoff: 12.0 Angstrom
- Training file: `P_GAP_20_fitting_data.xyz`
- Labels: `energy` and `forces`
- Validation split: 2%, seed 123
- Short-range cutoff: 4.5 Angstrom
- SWA stage: enabled from epoch 400

## Strengths

- Strong results on crystalline, 2D, and molecular-liquid phosphorus tests.
- Good ablation model for separating the effect of analytical dispersion from learned LES terms.

## Limitations

- Does not include the learned LES long-range channel.
- The GNB term is fitted and validated for elemental phosphorus in this training context, not for arbitrary mixed chemistry.
- Extrapolation to chemically different systems, impurities, adsorbates, or non-phosphorus species is unsupported.
