# Week 4 Final Status

## Scope

This note captures the current implementation state for the Week 4 final plan:

1. Retrain PhoBERT v1 on original `train_preprocessed.csv`
2. Run PhoBERT + LLM cascade on `WEAK_ASPECTS`

The goal is to preserve project state, actual code signatures, verified paths, and report-ready observations.

## Phase 0 Survey Summary

The codebase was inspected before making any edits, using the actual files in the current workspace.

### Actual Signatures

- `ABSAPhoBERT.__init__(model_name='vinai/phobert-base-v2', num_aspects=34, num_labels=4, dropout=0.2, encoder_option='concat_4_layers', focal_gamma=2.0)`
- `ABSAPhoBERT.forward(input_ids, attention_mask, labels=None, class_weights=None, token_type_ids=None)`
- `train(model, train_loader, dev_loader, class_weights, device, config, save_dir='outputs/models', results_dir='outputs/results', use_amp=False)`
- `run_epoch(model, dataloader, device, class_weights, optimizer=None, scheduler=None, grad_accum=1, is_train=True, use_amp=False, scaler=None)`
- `create_dataloaders(train_path, dev_path, test_path, tokenizer, batch_size=16, max_len=MAX_SEQ_LEN, num_workers=2, use_preprocessed=True)`
- `evaluate_predictions(y_true, y_pred, title='Evaluation', save_path=None, exclude_aspects=None)`
- `load_best_model(checkpoint_path, model, device)`
- `LLMClient.__init__(provider, api_key=None, model=None, cache_dir='outputs/llm_cache', max_retries=5, retry_delay=5.0)`
- `ABSARetriever.fit(train_df, text_col='processed_review')`
- `df_to_examples(df, indices)` is defined in `code/week3/icl_predictor.py`
- `build_prompt(test_review, examples, include_cot=True)`
- `parse_llm_output(raw_output)`
- `labels_dict_to_array(labels_dict)`

### Constants And Config

- `PHOBERT_MODEL_NAME` currently defaults to `PHOBERT_V2`
- `PHOBERT_V1 = 'vinai/phobert-base'`
- `PHOBERT_V2 = 'vinai/phobert-base-v2'`
- `WEAK_ASPECTS` already exists and contains 9 aspects
- `TRAIN_CONFIG` currently includes:
  - `learning_rate`
  - `lr_alpha`
  - `warmup_ratio`
  - `batch_size`
  - `grad_accumulation_steps`
  - `max_epochs`
  - `early_stop_patience`
  - `dropout`
  - `optimizer`
  - `scheduler`
  - `seed`
  - `max_seq_len`
  - `weight_clip`
  - `encoder_option`
  - `max_grad_norm`

### Notebook State

- `notebooks/week4_demo.ipynb` is now simplified for the final Kaggle run
- The notebook now keeps only the v2 cascade flow in the final section:
  - `Cell 15`: setup for PhoBERT v2 cascade
  - `Cell 16`: load v2 checkpoint and run a 10-review smoke test
  - `Cell 17`: run full cascade on the test set with RAG `k=4`
- Removed from the final notebook flow:
  - PhoBERT v1 retrain cell
  - PhoBERT v1 evaluation/comparison cell
  - PhoBERT v1 learning-curve cell

### Verified Baseline V2 Paths

- Best v2 checkpoint:
  - `outputs/results/week2_results_VNcoreNLP/models_cls_only/best_model.pt`
- Stored v2 test metrics:
  - `outputs/results/week2_results_VNcoreNLP/results_cls_only/week2_test_metrics.json`
- Verified stored metric:
  - `macro_combined_f1 = 0.5543128401889759`

## Additive Implementation State

### Cascade Module

`code/week3_part2/cascade_predictor.py` already exists in the current workspace and exposes the required Week 4 entry points:

- `WEAK_ASPECT_INDICES`
- `predict_single_phobert(...)`
- `predict_with_cascade(...)`
- `run_cascade_on_dataset(...)`
- `run_cascade_on_test(...)`

Current cascade logic:

- PhoBERT predicts all 34 aspects first
- Confidence is computed per aspect from max softmax probability
- Only aspects in `WEAK_ASPECTS` are considered for escalation
- If confidence on a weak aspect is below threshold, the pipeline:
  - retrieves RAG examples from train data
  - builds the LLM prompt
  - parses LLM output
  - overrides only selected weak-aspect predictions
- Safeguard currently active:
  - only override when the LLM label differs from PhoBERT
  - only override with non-`absent` LLM predictions
- Additional report-oriented tracking is now available in cascade stats:
  - `weak_trigger_counts`
  - `weak_override_counts`
  - optional `sample_records` when `return_records=True`

### Notebook Integration

No destructive notebook rewrite was needed because `week4_demo.ipynb` already contains the planned Week 4 cells and references:

- `phobert-base`
- `WEAK_ASPECTS`
- `cascade`

## Verification Completed

The following checks were run successfully in the current workspace:

- Constants import check: passed
- `cascade_predictor.py` syntax parse: passed
- `WEAK_ASPECT_INDICES` import check: passed
- `week4_demo.ipynb` content check: passed

## Result Tracking

### Verified Local Result

- PhoBERT v2 `cls_only + VnCoreNLP`
  - `ACD F1 = 0.6360`
  - `SPC F1 = 0.4727`
  - `Combined F1 = 0.5543`

### User-Reported Result

The current workspace does not yet contain `outputs/results/phobert_v1_cls_only/`, so the v1 result below is recorded as a user-reported experiment outcome and should be copied into saved artifacts when available:

- PhoBERT v1 `cls_only`
  - `ACD F1 = 0.5459`
  - `SPC F1 = 0.4189`
  - `Combined F1 = 0.4824`
  - `Weighted ACD F1 = 0.7463`

Interpretation:

- In the current project setting, PhoBERT v1 underperforms the verified v2 baseline
- This supports keeping PhoBERT v2 `cls_only + VnCoreNLP` as the main baseline for Week 4
- The remaining high-value path is selective LLM cascade on `WEAK_ASPECTS`

## Recommended Next Run Order

1. Keep PhoBERT v2 as the base classifier for cascade
2. Run notebook `Cell 18` first for a 10-review smoke test
3. Inspect:
   - `trigger_rate`
   - total overrides
   - whether overrides are concentrated on `WEAK_ASPECTS`
4. If smoke test is sane, run `Cell 19` on the full test set
5. Save for reporting:
   - overall cascade metrics
   - baseline vs cascade comparison
   - weak-aspect behavior
   - override rate

## Report Narrative

Suggested concise narrative for the final report:

- Re-running the paper-era backbone (`vinai/phobert-base`) did not improve results in the current implementation setting
- Data augmentation was tested twice and consistently degraded performance
- Therefore, the final improvement path focused on a hybrid cascade:
  - keep PhoBERT v2 as the strong baseline
  - escalate only uncertain weak-aspect cases to an LLM with RAG `k=4`
- This keeps API usage selective and ties the hybrid design directly to the observed failure modes of the classifier
