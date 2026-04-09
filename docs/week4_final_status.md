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

- `notebooks/week4_demo.ipynb` is now a final-only Kaggle notebook
- The notebook has been rewritten to 8 cells total:
  - markdown title/instructions
  - Kaggle GPU and dependency setup
  - clone/pull latest GitHub repo
  - load `OPENAI_API_KEY` from Kaggle Secrets
  - verify final-run files
  - setup PhoBERT v2 cascade
  - 10-review smoke test
  - full cascade test run
- Removed entirely from this notebook:
  - augmentation flow
  - explainability flow
  - PhoBERT v1 retrain/eval flow
  - old week4 branch-specific cells

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

Current cascade logic has been extended incrementally during Week 4:

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
- Extended uncertainty logic now supports:
  - `threshold`
  - `absent_threshold`
  - `min_non_absent_prob`
  - `margin_threshold`
- Extended cascade controls now support:
  - `override_only_from_absent`
  - `num_votes`
  - `llm_strategy`
- Two LLM modes now exist:
  - `free_json`
  - `candidate_rerank`

### Cascade Variants Implemented

The following non-destructive extensions were added to `code/week3_part2/cascade_predictor.py` during the Week 4 iteration cycle:

1. Selective escalation on `WEAK_ASPECTS`
2. Richer uncertainty rule for PhoBERT predictions
3. Target-only JSON prompt for weak-aspect override
4. Tracking fields for report analysis:
   - `weak_trigger_counts`
   - `weak_override_counts`
   - `empty_llm_responses`
   - `sample_records`
5. Optional `override_only_from_absent` mode
6. Optional `num_votes` self-consistency voting
7. Optional `llm_strategy='candidate_rerank'` mode
   - LLM is constrained to choose one of:
     - `keep_absent`
     - `positive`
     - `negative`
     - `neutral`
   - This is more controlled than the earlier free-form JSON override mode

### Notebook Integration

No destructive notebook rewrite was needed because `week4_demo.ipynb` already contains the planned Week 4 cells and references:

- `phobert-base`
- `WEAK_ASPECTS`
- `cascade`

Additional Kaggle notebooks were created for isolated experiment variants so earlier results remain reproducible:

- `notebooks/phobert-vncorenlp-llm-cascade-version1.2.ipynb`
- `notebooks/phobert-vncorenlp-llm-cascade-version1.3.ipynb`
- `notebooks/phobert-vncorenlp-llm-cascade-version1-4.ipynb`
- `notebooks/phobert-vncorenlp-llm-cascade-version1-5.ipynb`
- `notebooks/phobert-vncorenlp-llm-cascade-version1-6.ipynb`
- `notebooks/phobert-vncorenlp-llm-cascade-version1-7.ipynb`

Kaggle safety improvements were also added:

- notebook reloads `cascade_predictor` explicitly from disk
- notebook prints the loaded module path
- notebook prints `run_cascade_on_dataset(...)` signature before execution
- variant notebooks assert required new parameters exist before running

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

## Cascade Experiment Log

The cascade branch was explored through several controlled notebook variants. The table below is intended to be report-ready.

| Variant | Main Idea | Combined F1 | ACD F1 | SPC F1 | Key Notes |
| --- | --- | ---: | ---: | ---: | --- |
| PhoBERT v2 `cls_only` | Baseline classifier | 0.5543 | 0.6360 | 0.4727 | Verified local baseline |
| Cascade `v1.2` | Early cascade, `k=4`, weak-aspect override | 0.5538 | not recorded here as final reference | not recorded here as final reference | Trigger too low, only 4 overrides |
| Cascade `v1.3` | Improved uncertainty + target-only prompt | 0.5623 | 0.6261 | 0.4986 | First variant to beat baseline clearly |
| Cascade `v1.4` | Only override when PhoBERT predicts `absent` | 0.5570 | not used as final | not used as final | More conservative but worse than `v1.3` |
| Cascade `v1.5` | `k=8` + active top-5 weak aspects only | **0.5639** | around 0.6260 | around 0.5019 | Best hybrid result so far |
| Cascade `v1.6` | `v1.5` + vote-3 self-consistency | 0.5631 | 0.6248 | 0.5014 | Did not beat `v1.5` |
| Cascade `v1.7` | `v1.5` base + candidate rerank LLM mode | pending | pending | pending | Prepared for Kaggle run |

### Detailed Variant Notes

#### Cascade `v1.2`

- Base: PhoBERT v2 `cls_only`
- Retrieval: RAG `k=4`
- Behavior:
  - `trigger_rate = 0.07`
  - `llm_call_count = 42`
  - `total_overrides = 4`
- Interpretation:
  - Cascade barely changed predictions
  - The weakest aspects were not being triggered enough

#### Cascade `v1.3`

- Main changes:
  - broader uncertainty trigger
  - target-only JSON prompt
  - stronger weak-aspect focus
- Result:
  - `Combined F1 = 0.5623`
  - `trigger_rate = 0.285`
  - `llm_call_count = 171`
  - `total_overrides = 190`
  - `empty_llm_responses = 61`
- Interpretation:
  - This was the first successful hybrid improvement over the baseline
  - Improvement came mainly from better SPC on hard weak-aspect cases

#### Cascade `v1.4`

- Main change:
  - `override_only_from_absent = True`
- Result:
  - `Combined F1 = 0.5570`
  - `trigger_rate = 0.285`
  - `llm_call_count = 171`
  - `total_overrides = 213`
  - `empty_llm_responses = 50`
- Interpretation:
  - Restricting overrides only to absent predictions was too conservative
  - Better than baseline, but worse than `v1.3`

#### Cascade `v1.5`

- Main changes:
  - return to non-conservative override policy
  - increase retrieval to `k = 8`
  - reduce active weak aspects to top-5 with strongest signal:
    - `HOTEL#MISCELLANEOUS`
    - `FACILITIES#GENERAL`
    - `FACILITIES#COMFORT`
    - `FACILITIES#CLEANLINESS`
    - `FACILITIES#MISCELLANEOUS`
- Result:
  - `Combined F1 = 0.5639`
  - `trigger_rate = 0.2833`
  - `llm_call_count = 170`
  - `total_overrides = 212`
  - `empty_llm_responses = 46`
- Interpretation:
  - Best result in the entire Week 4 cascade branch
  - The two most effective tricks were:
    - `k = 8`
    - active top-5 weak-aspect filtering

#### Cascade `v1.6`

- Main changes:
  - same setup as `v1.5`
  - add `num_votes = 3` self-consistency voting
- Result:
  - `Combined F1 = 0.5631`
  - `Macro ACD F1 = 0.6248`
  - `Macro SPC F1 = 0.5014`
  - `trigger_rate = 0.2833`
  - `llm_call_count = 170`
  - `total_overrides = 223`
  - `empty_llm_responses = 44`
- Interpretation:
  - Slightly more stable formatting, but no score gain
  - More overrides did not translate into better final F1

#### Cascade `v1.7`

- Main changes:
  - keep the strong `v1.5` base configuration
  - add `llm_strategy = 'candidate_rerank'`
  - LLM no longer freely enumerates weak aspects
  - For each uncertain target aspect, LLM must choose exactly one of:
    - `keep_absent`
    - `positive`
    - `negative`
    - `neutral`
- Intended benefit:
  - reduce hallucinated free-form JSON behavior
  - make aspect-level decisions more controlled
  - test whether constrained reranking beats free-form override
- Status:
  - notebook prepared and pushed
  - result pending at the time of this note

## Current Best Result

### Best Baseline

- PhoBERT v2 `cls_only + VnCoreNLP`
  - `Combined F1 = 0.5543`

### Best Hybrid

- Cascade `v1.5`
  - `Combined F1 = 0.5639`

### Improvement Over Baseline

- Absolute gain from hybrid best over classifier baseline:
  - `0.5639 - 0.5543 = +0.0096`

This is the current strongest final result available in the Week 4 branch.

## Report-Ready Conclusions

1. Retraining PhoBERT v1 did not improve over the existing v2 baseline in this codebase and preprocessing setup.
2. Data augmentation had already been shown to hurt results in earlier experiments, so it was not pursued further.
3. The most effective Week 4 improvement was not changing the encoder, but selectively escalating difficult weak-aspect cases to an LLM with RAG.
4. The best-performing hybrid setup was:
   - PhoBERT v2 base classifier
   - selective cascade on top-5 weak aspects
   - RAG retrieval with `k = 8`
   - non-conservative override policy
5. Additional tricks beyond `v1.5` showed diminishing returns:
   - vote-3 self-consistency (`v1.6`) did not improve the score
   - candidate rerank (`v1.7`) was prepared as a final controlled experiment

## Important Residual Notes

- The checkpoint metadata printed during load sometimes showed `combined_f1 = 0.5451`, while the verified metrics JSON for the baseline is `0.5543`.
- For reporting, the metrics JSON should be treated as the authoritative baseline score.
- Rare aspects with extremely low support remained the hardest failure mode across all cascade variants.
- The strongest cascade gains came from a limited subset of weak aspects rather than from all 9 weak aspects equally.

## Recommended Next Run Order

1. Treat `v1.5` as the current best final hybrid checkpoint for reporting.
2. If continuing experiments, run `v1.7` as the last controlled Week 4 attempt.
3. Record after each run:
   - `Combined F1`
   - `Macro ACD F1`
   - `Macro SPC F1`
   - `trigger_rate`
   - `total_overrides`
   - `empty_llm_responses`
4. If `v1.7` does not beat `0.5639`, stop tuning and keep `v1.5` as the final hybrid result.
5. For the written report, include:
   - the ablation table above
   - the final comparison baseline vs hybrid best
   - 5-10 example overrides from `sample_records`

## Report Narrative

Suggested concise narrative for the final report:

- Re-running the paper-era backbone (`vinai/phobert-base`) did not improve results in the current implementation setting
- Data augmentation was tested twice and consistently degraded performance
- Therefore, the final improvement path focused on a hybrid cascade:
  - keep PhoBERT v2 as the strong baseline
  - escalate only uncertain weak-aspect cases to an LLM
  - progressively refine the hybrid policy through targeted ablations
- This keeps API usage selective and ties the hybrid design directly to the observed failure modes of the classifier
- The best final configuration was not the broadest or most complex cascade, but the most focused one:
  - RAG `k = 8`
  - only the top-5 weak aspects
  - selective override on uncertain cases
