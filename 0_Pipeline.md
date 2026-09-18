# General ML pipeline

```
                 ┌─────────────────┐
                 │  Business Goal  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Collection │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Validation │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Exploration│
                 │      (EDA)      │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Cleaning   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Data Splitting  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Feature         │
                 │ Engineering     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Baseline Model  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Selection │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Training        │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Hyperparameter  │
                 │ Tuning          │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Validation      │
                 └────────┬────────┘
                          ↓
                    Good enough?
                     /        \
                   No          Yes
                   │            │
                   └──→         ↓
                         Final Test
                              │
                              ↓
                       Model Packaging
                              │
                              ↓
                       Model Registry
                              │
                              ↓
                          Deployment
                              │
                              ↓
                           API/App
                              │
                              ↓
                         Monitoring
                              │
                              ↓
                     Retraining / Update
                              │
                              └──────→ Training
```
This is the general ML pipeline. Not every project needs every stage, but this architecture covers most ML projects.