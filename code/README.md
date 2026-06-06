# Zebrafish Heart Imaging

Analysis of cardiac development in zebrafish embryos using fluorescence microscopy and machine learning.

## Project Overview

This project studies heart tissue organization and actin cytoskeleton dynamics during zebrafish cardiac development under different experimental conditions (BDM treatment, Tricaine anesthesia, control).

### Datasets
- **cmlc2_lifeact_EGFP** — EGFP-tagged actin visualization
- **cmlc2_lifeactXnuclear** — Combined actin and nuclear staining
- **cmlc2_nuclear_dsRed** — Nuclear dsRed visualization

### Timepoints
- 32 hours post-fertilization (hpf)
- 48 hours post-fertilization (hpf)

## Directory Structure

```
zebrafish-heart-imaging/
├── data/
│   ├── raw/                 # Original microscopy data (.tif, .lif files)
│   └── processed/           # Preprocessed images and annotations
├── src/                     # Source code
│   ├── preprocessing/       # Image preprocessing scripts
│   ├── models/              # Model architectures
│   └── utils/               # Utility functions
├── models/                  # Trained model weights
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── tests/                   # Unit and integration tests
├── configs/                 # Configuration files (YAML/JSON)
├── logs/                    # Training and runtime logs
├── results/                 # Output results, visualizations, metrics
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd zebrafish-heart-imaging
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Processing
```bash
python src/preprocessing/process_images.py --config configs/preprocessing.yaml
```

### Model Training
```bash
python src/train.py --config configs/training.yaml
```

### Testing
```bash
pytest tests/
```

## Data Organization

Raw data is organized by:
- Experimental group (lifeact_EGFP, lifeactXnuclear, nuclear_dsRed)
- Developmental timepoint (32hpf, 48hpf)
- Treatment condition (BDM, TRICAINE, untreated)
- Individual image series (Series001.tif, etc.)

## Dependencies

See `requirements.txt` for full list of dependencies.

Key libraries:
- numpy
- scikit-image
- torch/tensorflow (for deep learning)
- matplotlib
- pandas

## Logging

Logs are stored in `logs/` directory with timestamps. Check here for:
- Training progress
- Processing errors
- Model evaluation metrics

## Results

Generated outputs are saved in `results/`:
- Visualizations (PNG/JPG)
- Metrics (CSV/JSON)
- Predictions
- Analysis reports

## Contributing

When contributing:
1. Create a new branch for your feature
2. Add tests in `tests/`
3. Update documentation as needed
4. Ensure all tests pass before submitting PR

## License

[Add your license here]

## Contact

[Add contact information]
