# 3Telegram
Different ideologies, same psychology
[![DOI](https://zenodo.org/badge/1170709980.svg)](https://doi.org/10.5281/zenodo.18970860)



The raw Telegram message data cannot be made publicly available due to the sensitive nature of the content. Aggregated analysis results, validation data, and analysis code are available and provided in the repository.

To run this experiment, you will need the full original data which can be made available upon reasonable request.

1. For topic modelling: Run steps 1–6 python scripts (in `topic_modelling/`)
2. To validate LLM annotations: Run `dcm_validation_batch.py`, then `dcm_validation_evaluate.py` (both in `dcm_annotations/`)
3. To run full annotations: Run `dcm_telegram_batch.py` (in `dcm_annotations/`)
4. For transfer entropy analysis: Run `dcm_transfer_entropy.py` (in `dcm_annotations/`)
5. For final stats: run and knit both `TM_analysis.Rmd` and `DCM_analysis.Rmd`

