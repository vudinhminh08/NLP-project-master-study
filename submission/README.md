# Submission Package

Submission includes:

- report
- code
- data
- notebooks results
- outputs
- demo app PhoBERT + LLM Explanation

## Important files

- Report: `Group16_Report_v12.docx`
- Best PhoBERT notebook:
  `notebooks/phobert_vnccorenlp_version3_clean_done(best_results).ipynb`
- Best PhoBERT checkpoint:
  `outputs/results/phobert_results/models_cls_only/best_model.pt`

## Main Results

- PhoBERT `cls_only`
- Test ACD F1: `0.6849`
- Test SPC F1: `0.5587`
- Test Combined F1: `0.6218`

## Run demo app

From `submission` directory:

```bash
python3 -m venv .venv-app
source .venv-app/bin/activate
pip install --upgrade pip
pip install -r app/requirements-app.txt
streamlit run app/streamlit_app.py
```

Demo app use:

```text
outputs/results/phobert_results/models_cls_only/best_model.pt
```

And VnCoreNLP directory:

```text
vncorenlp
```

If OpenAI API key is missing, PhoBERT prediction is still the core component, but explanation will not run.

