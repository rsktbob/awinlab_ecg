# Refactoring log for ecg_tool

## [2026-02-27] Refactored `ecg_tool.py`
- Refactored `ecg_tool.py` to improve modularity and readability.
- Consolidated single-label and multi-label evaluation into a single `evaluate` function with wrappers for backward compatibility.
- Improved `prepare_data` with constants and better path handling.
- Refined `train_model` with a more generic gradient setting utility and progressive unfreezing.
- Added type hints and docstrings to all major functions.
- Introduced `create_ecg_model` for centralized model architecture definition.
- Fixed an inconsistency in `load_model` return values (now returns `model, class_names`).
- Cleaned up redundant code.

## [2026-02-27] Refactored Evaluation Functions
- Unified `evaluate_singlelabel` and `evaluate_multilabel` into a common internal evaluation logic.
- Improved code reuse and added better error handling for AUROC calculation.
- Maintained backward compatibility via wrappers.

