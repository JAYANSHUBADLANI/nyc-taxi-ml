# Project Progress

## Done
- Set up directory structure.
- Created `README.md`, `.gitignore`, and `PROGRESS.md`.
- Initiated environment setup (Java and PySpark installation).
- Phase 0: Environment & Smoke Test
  - Generated a bounded gold subset of NYC TLC taxi data (~3M rows for Jan 2023).
  - Executed end-to-end tiny-slice smoke test successfully.
- Phase 1: Feature Engineering & Baseline
  - Built PySpark transformers for temporal and categorical features.
  - Used `FeatureHasher` for `zone_pair` to avoid sparsity OOM errors.
  - Calculated naive baseline predictor (RMSE 12.10).
- Phase 2: Spark ML Pipeline (Full Scale)
  - Trained `RandomForestRegressor` over full 3 million rows.
  - Held out final 20% by time correctly avoiding data leakage.
- Phase 3: Tuning & Evaluation
  - Tuned with `TrainValidationSplit`.
  - Evaluated the final model (RMSE 4.79).
  - Showed error breakdowns by borough and hour of the day.
  - Passed reproducibility check precisely.
- Phase 4: Documentation
  - Finished `README.md` with complete metrics, honest assessment, and local paths removed.

## Pending
- None. The project runs end to end and is ready to push.
