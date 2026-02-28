# Colab-Equivalent Before/After DICOM App (Streamlit)

This Streamlit app is rewritten to follow the same pipeline style as your Colab code:

- `before` pipeline using healthy references
- `after` pipeline using healthy references + AE + neurosymbolic cleanup
- laser treatment simulation with GIF output

## Folder layout required

Place healthy reference DICOM folders in the repo:

- `data/trial1`
- `data/trial2`
- `data/trial3`

You upload only the cancer case in the UI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Usage

1. Ensure `data/trial1..3` contain healthy DICOM files.
2. Launch app and keep sidebar defaults unless needed.
3. Upload one cancer DICOM case (files or zip).
4. Click `Run Comparison`.
5. Download:
   - `before_treatment.gif`
   - `after_treatment.gif`

## Outputs

Each run writes:

- `outputs/<timestamp>/before/before_treatment.gif`
- `outputs/<timestamp>/after/after_treatment.gif`
- `outputs/<timestamp>/model_output.json`
