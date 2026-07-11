# Model Comparison Report

> Both models were trained with **linear probing (frozen backbone)** — see PHASE6_LOG.md. This was a VRAM-safety choice for the 2GB MX550 GPU, not an inherent architecture limitation. Full fine-tuning could narrow or reverse the gap below.

| Metric | efficientnet | vit |
|---|---|---|
| Accuracy | 0.9735 | 0.9306 |
| Precision (macro) | 0.9640 | 0.9201 |
| Recall (macro) | 0.9687 | 0.9276 |
| F1-score (macro) | 0.9656 | 0.9216 |
| ROC-AUC (macro, OvR) | 0.9998 | 0.9991 |
| Inference time (ms/image) | 4.941 | 5.519 |
| Model size (MB) | 15.77 | 21.16 |
| Total parameters | 4,056,226 | 5,531,750 |
| Trainable parameters | 48,678 | 7,334 |

**Best model by macro-F1:** `efficientnet` (macro_f1=0.9656) — macro-F1 is used as the selection criterion (not accuracy) due to the 36.23x class imbalance confirmed in Phase 4.
