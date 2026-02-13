# TSE training pipeline (iTransformer)

This folder provides a **multi-step** training example to adapt the TSE workflow to time-series forecasting:

1. Train parent A with Adam.
2. Train parent B with SAM/ASAM.
3. Run TSE (SAM + Fisher-weighted twin EWC regularization toward parent A/B).
4. Optionally iterate with new parents.

Run:

```bash
bash scripts/tse/train_tse_pipeline.sh
```

> The script contains placeholder checkpoint paths for step 3. Update them to your real checkpoint outputs in `./checkpoints/`.
