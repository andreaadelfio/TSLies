# TSLies - Time Series Anomalies Detection Framework

## What is it?
**TSLies** (Time Series anomaLIES) is an advanced anomaly detection framework using state-of-the-art Machine Learning techniques and the Poisson-FOCuS triggering algorithm for real-time anomaly detection in time series data. This framework provides a comprehensive suite of ML models, from deterministic to Bayesian approaches, for robust background modeling and anomaly detection in any time series dataset.

## Table of contents
- [Main Features](#main-features)
- [Architecture Overview](#architecture-overview)
- [Machine Learning Models](#machine-learning-models)
- [Installation and Dependencies](#installation-and-dependencies)
- [Modules](#modules)
- [Scripts and Pipelines](#scripts-and-pipelines)
- [Usage](#usage)
- [Data Structure](#data-structure)
- [Contributing](#contributing)
- [Contact](#contact)

## Main Features

This repository provides a comprehensive framework for anomaly detection in time series data:

### Core Capabilities
- **Real-time Background Modeling**: Multiple ML architectures for background prediction
- **Bayesian Uncertainty Quantification**: Probabilistic models for reliable anomaly detection
- **Spectral Domain Learning**: Frequency-domain neural networks
- **FOCuS Change Point Detection**: Optimal changepoint detection algorithm
- **Multi-dataset Validation**: Cross-referencing with external event catalogs and metadata
- **Automated Visualization**: Scientific plotting with LaTeX formatting

### Key Innovations
- **Hybrid Model Ensemble**: Combination of deterministic and Bayesian approaches
- **Modular Architecture**: Individual files for each ML model type for maximum maintainability
- **Scalable Pipeline Architecture**: Modular design for large-scale data processing
- **Advanced Uncertainty Handling**: Critical for low false-positive anomaly detection

## Architecture Overview

The framework follows a modular, three-stage pipeline architecture suitable for any time series anomaly detection task:

```mermaid
flowchart TD
    A[Raw Time Series Data] --> B[Data Preprocessing]
    C[External Features - if any] --> B
    D[Contextual Data - if any] --> B
    B --> E[Feature Engineering]
    E --> F[Background Modeling]
    F --> G[Anomaly Detection]
    G --> H[Catalog Validation]
    H --> I[Scientific Visualization]
```

### Data Flow
1. **Input Integration**: Time series data, external features, contextual information
2. **Background Prediction**: ML models predict normal system behavior
3. **Anomaly Detection**: FOCuS algorithm identifies deviations from background
4. **Validation**: Cross-reference detections with known event catalogs or external metadata
5. **Analysis**: Generate scientific plots and performance metrics

## Machine Learning Models

### Deterministic Models
- **FFNNPredictor**: Feed-Forward Neural Network for baseline background modeling
- **SpectralDomainFFNNPredictor**: Frequency-domain Neural Network using FFT-based loss functions
- **RNNPredictor**: Recurrent Neural Network for temporal dependencies

### Bayesian Models (with Uncertainty Quantification)
- **BNNPredictor**: Bayesian Neural Network with variational inference
- **PBNNPredictor**: Probabilistic Bayesian Neural Network with enhanced uncertainty
- **ABNNPredictor**: Advanced Bayesian Neural Network with tensorflow-probability 
- **MCMCBNNPredictor**: MCMC-based Bayesian Neural Network for full posterior sampling (to be finished)

### Non-Parametric Models
- **MedianKNeighborsRegressor**: K-Nearest Neighbors with median aggregation
- **MultiMeanKNeighborsRegressor**: Multi-output K-NN for baseline comparisons

### Model Selection Criteria
- **Uncertainty Quantification**: Essential for reliable anomaly detection
- **Computational Efficiency**: Real-time processing requirements
- **Interpretability**: Understanding model decisions and feature importance

## Installation and Dependencies

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for neural networks training)

### Quick Installation
```bash
git clone https://github.com/andreaadelfio/TSLies.git
cd TSLies
pip install -e .
```

Or install from PyPI (not yet available):
```bash
pip install tslies
```

### Python Dependencies
Core packages include:
- `tensorflow>=2.16.0` (neural networks)
- `tensorflow-probability>=0.23.0` (Bayesian models)
- `tf-keras>=2.16.0` (Keras integration)
- `pandas>=1.3.0` (data manipulation)
- `scikit-learn>=1.0.0` (classical ML)
- `matplotlib>=3.5.0` (visualization)
- `seaborn>=0.11.0` (advanced plotting)
- `numpy>=1.21.0` (numerical computing)
- `astropy>=5.0.0` (astronomical data formats)

## Modules

TSLies is organized into a modular architecture separating generic time series functionality from domain-specific applications:

### Core Modules (`modules/`)

#### **modules/config.py**
Centralized configuration management:
- File paths and directory structure
- Configurable thresholds and parameters

#### **modules/background/**
Complete ML model ecosystem with modular architecture:
- `mlobject.py`: Base class with common ML functionality
- `losses.py`: Custom loss functions (spectral, Bayesian NLL)
- `ffnnpredictor.py`: Feed-Forward Neural Network predictor
- `rnnpredictor.py`: Recurrent Neural Network for temporal dependencies
- `bnnpredictor.py`: Bayesian Neural Network with uncertainty quantification
- `pbnnpredictor.py`: Probabilistic Bayesian Neural Network
- `abnnpredictor.py`: Approximate Bayesian Neural Network
- `mcmcbnnpredictor.py`: MCMC-based Bayesian Neural Network
- `spectraldomainffnnpredictor.py`: Frequency-domain neural network
- `knnpredictors.py`: K-Nearest Neighbors regressors (median/mean variants)
- Automated hyperparameter optimization and model persistence

#### **modules/trigger.py**
Advanced anomaly detection:
- FOCuS-Gaussian and FOCuS-Poisson algorithm implementation (Kester Ward, 2021)
- Z-score detection
- Multi-variate time series trigger merging
- Temporal clustering and filtering

#### **modules/plotter.py** 
Scientific visualization suite:
- Automated anomaly plotting
- LaTeX-formatted scientific notation
- Multi-panel time series with residuals
- Export-ready figures for pubblications

#### **modules/utils.py**
Essential utilities:
- Data manipulation and masking
- Time series processing
- Logging and debugging
- File I/O operations

#### **modules/dataset.py**
Primary data processing:
- Raw data file parsing and conversion
- Multi-channel data handling
- Temporal binning and aggregation

### Domain-Specific Applications (`applications/`)

#### **applications/acd/**
ACD (Anti-Coincidence Detector) specific modules:

- **spacecraft.py**: Spacecraft parametric data integration
- **solar.py**: Solar environmental monitoring data integration  
- **catalogs.py**: Event catalog cross-referencing and validation
- **main_*.py**: Complete analysis pipelines for ACD data

#### **applications/intesa sanpaolo/**
Intesa Sanpaolo application:

- **main_intesa.py**: analysis pipelines for Intesa Sanpaolo data

## Usage

### Quick Start with TSLies

1. **Import core TSLies modules**:
   ```python
   # Import core TSLies components
   from modules.config import DIR, BACKGROUND_PREDICTION_FOLDER_NAME
   from modules.background import FFNNPredictor, BNNPredictor
   from modules.trigger import Trigger
   from modules.plotter import Plotter
   ```

2. **Train a background model**:
   ```python
   # Create and train a neural network background model
   model = FFNNPredictor(df_data, y_cols, x_cols, y_cols_raw, y_pred_cols, y_smooth_cols)
   model.set_hyperparams(params)
   model.create_model()
   history = model.train()
   ```

3. **Run anomaly detection**:
   ```python
   # Apply FOCuS algorithm for changepoint detection
   trigger = Trigger(tiles_df, y_cols, y_pred_cols, y_cols_raw, units, latex_y_cols)
   anomalies, significance_df = trigger.run(thresholds, type='focus')
   ```

4. **Visualize results**:
   ```python
   # Generate scientific plots
   plotter = Plotter(df=anomalies)
   plotter.plot_anomalies_in_catalog(trigger_type, support_vars, thresholds, tiles_df, y_cols, y_pred_cols)
   ```

### ACD-Specific Application

For ACD (Anti-Coincidence Detector) specific analysis:

```python
# Import ACD-specific modules
from applications.acd.spacecraft import SpacecraftOpener
from applications.acd.solar import SunMonitor  
from applications.acd.catalogs import CatalogReader

# Or run complete ACD pipelines
from applications.acd import main_ml, main_trigger, main_acd
```

### Advanced Usage

#### Model Comparison
```python
# Compare multiple ML architectures
from modules.background import (
    FFNNPredictor, 
    BNNPredictor, 
    SpectralDomainFFNNPredictor,
    RNNPredictor
)

models = [FFNNPredictor, BNNPredictor, SpectralDomainFFNNPredictor, RNNPredictor]
results = {}

for ModelClass in models:
    model = ModelClass(df_data, y_cols, x_cols, ...)
    model.create_model()
    history = model.train()
    results[ModelClass.__name__] = model.evaluate()
```

## Data Structure

### Required Directory Layout
```
TSLies/
├── results/                 # Model outputs and analysis
│   └── YYYY-MM-DD/
│       ├── background_prediction/
│       └── trigger_results/
├── logs/                   # System logs and debugging
├── modules/               # Core TSLies framework
├── applications/         # Domain-specific applications
│   └── acd/             # ACD-specific modules and scripts
└── pipelines/           # End-to-end workflows
```

## Contributing

We welcome contributions to this Time Series Anomaly Detection Framework!
## Contact

**Lead Developer**: Andrea Adelfio  
**Institution**: INFN (Istituto Nazionale di Fisica Nucleare)  
**Email**: 
- <and.adelfio@gmail.com>
- <andrea.adelfio@pg.infn.it>

**Project Status**: Active development

### Acknowledgments
- **FOCuS Algorithm**: Ward (2021)
- **Research Community**: Open-source machine learning libraries and frameworks
- **Computing Resources**: High-performance computing support

---

### Citation
If you use TSLies in your research, please cite:
```
@software{adelfio2024tslies,
  author = {Adelfio, Andrea},
  title = {TSLies: Time Series Anomaly Detection Framework},
  url = {https://github.com/andreaadelfio/TSLies},
  year = {2024},
  institution = {INFN}
}
```

<hr>

[Go to Top](#table-of-contents)
