# Cancer DICOM Before/After GIF App (Streamlit)

This app uses one cached model file only:

- `model_cache/after_ae_model.keras`

You upload one cancer DICOM case (files or zip), and the app generates:

- `before_treatment.gif`
- `after_treatment.gif`

No healthy trial uploads are required.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Required model file

Place your model at:

- `model_cache/after_ae_model.keras`

If the model file is missing, the app will stop with an error.

## Usage

1. Launch the app.
2. Confirm model path is `model_cache/after_ae_model.keras`.
3. Upload one cancer DICOM case.
4. Click `Generate Before/After GIFs`.
5. Download both GIF files from the UI.

## Outputs

Each run saves under:

- `outputs/<timestamp>/before_treatment.gif`
- `outputs/<timestamp>/after_treatment.gif`
- `outputs/<timestamp>/model_output.json`
