# Production-oriented ML project structure

```
my-ml-project/
│
├── configs/                         # ⚙️ All project configuration/settings
│   │
│   ├── config.yaml                 # Global settings: project name, seed, device, paths, logging
│   ├── data.yaml                   # Dataset settings: paths, target, split, sequence length, etc.
│   ├── model.yaml                  # Model architecture settings: layers, hidden size, dropout, etc.
│   ├── training.yaml               # Training settings: epochs, batch size, LR, optimizer, scheduler
│   └── tuning.yaml                 # Hyperparameter tuning: search space, trials, objective, Optuna
│
├── data/                            # 📊 All datasets used by the project
│   │
│   ├── raw/                        # Original/unmodified data downloaded or collected
│   │                                # NEVER modify original source data here
│   │
│   ├── interim/                    # Temporary/intermediate data during preprocessing
│   │                                # Used between raw and final processed data
│   │
│   ├── processed/                  # Cleaned/transformed data ready for ML training
│   │                                # Example: train.csv, validation.csv, test.csv
│   │
│   └── external/                   # Data obtained from external APIs, databases, or sources
│
├── notebooks/                       # 📓 Jupyter notebooks for exploration and experimentation
│   │
│   ├── 01_exploration.ipynb        # Understand dataset: shape, columns, missing values, duplicates
│   ├── 02_data_analysis.ipynb      # EDA: distributions, correlations, patterns, outliers
│   ├── 03_feature_engineering.ipynb# Experiment with creating/selecting useful features
│   ├── 04_model_experiment.ipynb   # Experiment with different models/architectures
│   └── 05_evaluation.ipynb         # Analyze model results, metrics, errors, and visualizations
│
├── src/                             # 🧠 Main reusable ML application/source code
│   │
│   ├── __init__.py                 # Makes src a Python package
│   │
│   ├── config/                     # Configuration loading and management
│   │   │
│   │   ├── __init__.py             # Makes config a Python package
│   │   └── loader.py               # Reads YAML configuration and makes it available to Python
│   │
│   ├── data/                       # 📥 Data loading, validation, splitting, and dataset handling
│   │   │
│   │   ├── __init__.py             # Makes data a Python package
│   │   ├── loader.py               # Loads data from CSV, JSON, images, DB, API, etc.
│   │   ├── validator.py            # Checks data quality, schema, missing/invalid values
│   │   ├── splitter.py             # Splits data into train, validation, and test sets
│   │   └── dataset.py              # Converts data into ML/DL Dataset/DataLoader format
│   │
│   ├── preprocessing/              # 🧹 Converts raw data into ML-ready input
│   │   │
│   │   ├── __init__.py             # Makes preprocessing a Python package
│   │   ├── cleaner.py               # Cleans data: missing values, duplicates, invalid records
│   │   ├── transformer.py           # Scaling, normalization, encoding, padding, transformations
│   │   └── tokenizer.py             # NLP tokenization and vocabulary processing
│   │
│   ├── features/                   # 🔧 Feature engineering
│   │   │
│   │   ├── __init__.py             # Makes features a Python package
│   │   └── engineer.py             # Creates, transforms, selects, or removes ML features
│   │
│   ├── models/                     # 🤖 Model definition and model management
│   │   │
│   │   ├── __init__.py             # Makes models a Python package
│   │   ├── architecture.py         # Defines neural network/model architecture
│   │   ├── factory.py              # Creates different model types from configuration
│   │   └── checkpoint.py           # Saves, loads, and manages model checkpoints
│   │
│   ├── training/                   # 🎯 Everything related to model training
│   │   │
│   │   ├── __init__.py             # Makes training a Python package
│   │   ├── trainer.py              # Main training loop: forward → loss → backward → optimizer
│   │   ├── loss.py                 # Loss function definitions: MSE, CrossEntropy, etc.
│   │   ├── optimizer.py             # Optimizer configuration: Adam, AdamW, SGD, etc.
│   │   └── scheduler.py             # Learning-rate schedulers and LR adjustment
│   │
│   ├── tuning/                     # 🔍 Hyperparameter optimization
│   │   │
│   │   ├── __init__.py             # Makes tuning a Python package
│   │   ├── objective.py            # Defines what one tuning trial should do and return
│   │   └── tuner.py                # Runs hyperparameter search using Optuna/other framework
│   │
│   ├── evaluation/                 # 📈 Model performance evaluation
│   │   │
│   │   ├── __init__.py             # Makes evaluation a Python package
│   │   ├── metrics.py              # Defines metrics: accuracy, F1, MAE, RMSE, etc.
│   │   ├── evaluator.py            # Runs model evaluation on validation/test data
│   │   └── error_analysis.py       # Investigates wrong predictions and model weaknesses
│   │
│   ├── inference/                  # 🔮 Prediction logic after model training
│   │   │
│   │   ├── __init__.py             # Makes inference a Python package
│   │   └── predictor.py            # Loads trained model and generates predictions
│   │
│   └── utils/                      # 🛠️ Small reusable helper functions
│       │
│       ├── __init__.py             # Makes utils a Python package
│       ├── seed.py                 # Sets random seeds for reproducible experiments
│       ├── logger.py               # Logging configuration and log handling
│       └── device.py               # Selects CPU, CUDA/GPU, or Apple MPS device
│
├── scripts/                        # ▶️ Entry-point scripts used to run ML operations
│   │
│   ├── prepare_data.py             # Runs complete data preparation pipeline
│   ├── train.py                    # Starts model training
│   ├── tune.py                     # Starts hyperparameter tuning
│   ├── evaluate.py                # Runs final model evaluation
│   └── predict.py                 # Runs prediction from command line
│
├── models/                          # 💾 Trained model files/artifacts
│   │
│   ├── checkpoints/                # Intermediate model checkpoints during training
│   ├── best/                       # Best model selected according to validation metric
│   └── production/                 # Model currently approved/selected for production
│
├── experiments/                    # 🧪 History of different ML experiments
│   │
│   ├── experiment_001/             # First experiment: config + metrics + model/results
│   ├── experiment_002/             # Second experiment
│   └── experiment_003/             # Third experiment
│
├── outputs/                         # 📤 Results generated by training/evaluation
│   │
│   ├── metrics/                   # Accuracy, loss, F1, MAE, RMSE, etc.
│   ├── plots/                     # Loss curves, confusion matrix, graphs, charts
│   ├── predictions/               # Model prediction files/results
│   └── logs/                      # Training, tuning, evaluation logs
│
├── tests/                           # 🧪 Automated tests for the ML application
│   │
│   ├── test_data.py               # Tests data loading, validation, and splitting
│   ├── test_preprocessing.py      # Tests cleaning, transformation, tokenization
│   ├── test_model.py              # Tests model creation, input/output shapes, forward pass
│   ├── test_training.py           # Tests training logic and training components
│   └── test_inference.py          # Tests prediction/inference functionality
│
├── api/                             # 🌐 API layer for serving the trained model
│   │
│   ├── __init__.py                # Makes api a Python package
│   ├── main.py                    # FastAPI application and API endpoints
│   └── schemas.py                 # Request/response validation schemas
│
├── .env                             # 🔐 Environment variables and secrets
│                                    # Example: API keys, database URL, environment settings
│
├── .gitignore                       # 🚫 Files/folders Git should not track
│
├── pyproject.toml                   # 📦 Python project configuration and dependencies
│                                    # Especially important when using uv
│
├── README.md                        # 📖 Project documentation and usage instructions
│
└── uv.lock                          # 🔒 Exact locked Python dependency versions for reproducibility
```

## The easiest way to remember the whole structure
```
my-ml-project/
│
├── configs/       → ⚙️ WHAT settings do I use?
│
├── data/          → 📊 WHAT data do I use?
│
├── notebooks/     → 📓 WHAT am I exploring?
│
├── src/           → 🧠 HOW does my ML system work?
│
├── scripts/       → ▶️ HOW do I run the system?
│
├── models/        → 💾 WHAT trained models do I have?
│
├── experiments/   → 🧪 WHAT have I tried?
│
├── outputs/       → 📤 WHAT results did I get?
│
├── tests/         → ✅ DOES everything work?
│
└── api/           → 🌐 HOW does the application use the model?
```
### And inside src/:
```
src/
│
├── config/         → Load configuration
├── data/           → Get and prepare data
├── preprocessing/  → Clean/transform data
├── features/       → Create features
├── models/         → Define model
├── training/       → Train model
├── tuning/         → Find best hyperparameters
├── evaluation/     → Measure model
├── inference/      → Make predictions
└── utils/          → Common helpers
```

