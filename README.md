# Before/After DICOM Comparator (Streamlit)

This app takes one uploaded cancer DICOM case, runs two pipelines (`before` and `after`), and renders:

- side-by-side GIFs for visual comparison
- side-by-side metrics table
- cached `after` model loading so retraining is not repeated every run

## Features

- Streamlit upload UI for cancer DICOM files (`.dcm`) or a `.zip`.
- `before` and `after` pipelines run on the same upload.
- `after` pipeline supports model caching at `model_cache/after_ae_model.keras`.
- optional `Load Saved model_output.json` mode to render previous results without rerunning.
- in-app model download button for the cached `.keras` file.
- per-run output bundle saved under `outputs/<timestamp>/`.

## Repository structure

- `app.py`: Streamlit app + pipeline execution.
- `requirements.txt`: runtime dependencies.
- `runtime.txt`: Streamlit Cloud Python runtime pin (`python-3.11`).
- `.streamlit/config.toml`: Streamlit runtime settings.
- `.github/workflows/ci.yml`: lightweight CI checks.
- `.gitattributes`: Git LFS tracking for `.keras` model files.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Put your trained model at `model_cache/after_ae_model.keras`.
2. Upload one cancer case (DICOM files or zip).
3. Keep cached model path as `model_cache/after_ae_model.keras`.
4. Keep `Reuse cached AFTER model only (no retraining)` enabled to avoid retraining on cloud.
5. (Optional) Add healthy folder paths in the sidebar only if you want:
   - `BEFORE` pipeline, and/or
   - AFTER healthy-diff scoring in addition to model scoring.
6. For Streamlit Cloud stability, keep:
   - `Generate GIFs` off unless needed
   - `Max slices to process per run` around `80-150`
   - lower `AFTER model train epochs` if retraining is enabled
7. Click **Run Comparison**.

If retraining is enabled and healthy paths are provided:
- the app trains and saves `model_cache/after_ae_model.keras`.

With cache-only mode:
- the app loads cached model and skips retraining.

Alternative (no rerun):
- switch to `Load Saved model_output.json`, upload a prior run JSON, and render stored before/after outputs.

## Outputs

Each run creates:

- `outputs/<timestamp>/before/before_treatment.gif`
- `outputs/<timestamp>/after/after_treatment.gif`
- `outputs/<timestamp>/before/slice_detected_counts.csv`
- `outputs/<timestamp>/after/slice_detected_counts.csv`
- `outputs/<timestamp>/model_output.json`

## GitHub model storage

This repo includes `.gitattributes` for Git LFS on `*.keras`.

Use:

```bash
git lfs install
git add .gitattributes
git add model_cache/after_ae_model.keras
git commit -m "Add cached AFTER model"
git push
```

If the model file is small and you prefer normal Git, remove the `*.keras` LFS rule from `.gitattributes`.

## Non-clinical disclaimer

This is a research/prototype workflow and must not be used for clinical decision-making.
