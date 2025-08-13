# RSNA Intracranial Aneurysm Detection

Kaggle Competition: [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

## Overview

This repository contains code for detecting and localizing intracranial aneurysms in brain imaging data (CTA/MRA/MRI) using deep learning approaches.

### Competition Goal
- **Primary Objective**: Maximize weighted multilabel AUC ROC
- **Key Metric**: "Aneurysm Present" classification (weight=13)
- **Secondary**: Classify 13 anatomical locations

## Project Structure

```
kaggle-rsna/
├── kaggle_evaluation/     # Kaggle evaluation API implementation
│   ├── core/              # Core evaluation modules
│   ├── rsna_gateway.py    # Test data iteration handler
│   └── rsna_inference_server.py  # Main inference server
├── file.py                # DICOM loading utilities
├── prompt.MD              # Competition guidelines
├── CLAUDE.md              # AI assistant documentation
└── notebooks/             # Jupyter notebooks for exploration
```

## Data

**Note**: Large data files (250GB+) are not included in this repository. You need to download them from Kaggle:

- Training data: ~3,000 CTA/MRA/MRI series
- Test data: Accessed via Kaggle evaluation API
- Segmentations: Vessel masks for subset of training data
- Localizers: Weak supervision annotations

## Labels (14 Binary Targets)

1. Left/Right Infraclinoid Internal Carotid Artery
2. Left/Right Supraclinoid Internal Carotid Artery  
3. Left/Right Middle Cerebral Artery
4. Anterior Communicating Artery
5. Left/Right Anterior Cerebral Artery
6. Left/Right Posterior Communicating Artery
7. Basilar Tip
8. Other Posterior Circulation
9. **Aneurysm Present** (weight=13)

## Setup

### Requirements
```bash
pip install pydicom numpy polars
# Additional dependencies for model training
pip install torch torchvision nibabel scikit-learn
```

### Data Download
1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data)
2. Extract to appropriate directories following .gitignore structure

## Usage

### DICOM Loading
```python
from file import load_series_to_volume
from pathlib import Path

series_path = Path("kaggle_evaluation/series/YOUR_SERIES_ID")
volume, spacing, description = load_series_to_volume(series_path)
```

### Local Testing
```bash
python -m kaggle_evaluation.rsna_inference_server
```

## Approach

- **Data Processing**: DICOM to numpy conversion with HU normalization
- **Model Architecture**: 3D CNN or 2.5D hybrid for volumetric classification
- **Training Strategy**: Patient-level stratified k-fold cross-validation
- **Evaluation**: Weighted AUC ROC with ensemble predictions

## Competition Constraints

- No internet access during inference
- Strict CPU/GPU runtime limits
- Deterministic/reproducible results required
- Process one series at a time via evaluation API

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for educational and competition purposes. Please refer to Kaggle competition rules for usage restrictions.

## Acknowledgments

- RSNA (Radiological Society of North America)
- Kaggle competition organizers
- Medical imaging community