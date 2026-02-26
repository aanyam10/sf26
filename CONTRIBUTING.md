# Contributing

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

## Pull requests

1. Create a branch from `main`.
2. Keep changes scoped to one objective.
3. Ensure code compiles:
   - `python3 -m py_compile app.py`
4. Update `README.md` when behavior or setup changes.
5. Open a PR with:
   - what changed
   - why
   - test evidence

## Model files

- `*.keras` is tracked via Git LFS by default.
- If your environment does not use LFS, adjust `.gitattributes` before committing model artifacts.
