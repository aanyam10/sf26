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

1. In sidebar, choose healthy input mode:
   - `Bundled trial folders (repo)` (no healthy uploads each run), or
   - `Upload healthy DICOM files` (recommended for Streamlit Cloud if healthy folders are not bundled), or
   - `Local folder paths` (for local runs on your machine).
2. Upload one cancer case (DICOM files or zip).
3. If using upload mode, upload healthy trials for `trial1`, `trial2`, `trial3` in the Run tab.
4. Keep cached model path as `model_cache/after_ae_model.keras`.
5. Keep `Reuse cached AFTER model only (no retraining)` enabled to avoid retraining on cloud.
6. For Streamlit Cloud stability, keep:
   - `Generate GIFs` off unless needed
   - `Max slices to process per run` around `80-150`
   - lower `AFTER model train epochs` if retraining is enabled
7. Click **Run Comparison**.

First run:
- trains and saves `model_cache/after_ae_model.keras`.

Later runs:
- loads cached model and skips retraining.

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
