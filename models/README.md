# OurionSpectra trained model assets

- `ourion_composition_model.pth` — connected atmospheric-composition network (104 → 128 → 64 → 32 → 5).
- `spectra_train.csv` — exact training spectra used to reproduce the input StandardScaler.
- `fm_parameter_train.csv` — exact training parameters used to reproduce the output StandardScaler.
- `ourion_flux_model.pth` — uploaded legacy 2-input flux regressor retained as an asset. It is **not** used in default recovery because its training domain is physical flux units and its wavelength/flux scaling does not match the application's normalized recovery interface.
