# General ML pipeline

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
That is the general ML pipeline. Not every project needs every stage, but this architecture covers most ML projects.