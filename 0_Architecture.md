

```
                    ┌─────────────────────┐
                    │       DATA          │
                    │ Raw / External      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DATA VALIDATION     │
                    │ Quality / Schema    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PREPROCESSING       │
                    │ Clean / Transform   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FEATURE ENGINEERING │
                    │ Features / Encoding │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DATA SPLITTING      │
                    │ Train / Val / Test  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ MODEL               │
                    │ Architecture        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ TRAINING            │
                    │ Optimizer / LR      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ HYPERPARAMETER      │
                    │ TUNING              │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ EVALUATION          │
                    │ Metrics / Errors    │
                    └──────────┬──────────┘
                               │
                         Pass validation?
                          /            \
                        NO              YES
                        │                │
                        ▼                ▼
                  Improve model      Save model
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ MODEL REGISTRY      │
                              │ Version / Metadata  │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ INFERENCE           │
                              │ Prediction          │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ API / APPLICATION   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ MONITORING          │
                              │ Drift / Performance │
                              └─────────────────────┘
```