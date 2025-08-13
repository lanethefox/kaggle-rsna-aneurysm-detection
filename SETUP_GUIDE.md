# RSNA Intracranial Aneurysm Detection - Setup & Usage Guide

## Table of Contents
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Library Usage](#library-usage)
- [Evaluation API](#evaluation-api)
- [Model Development](#model-development)
- [Submission Guide](#submission-guide)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training, optional)
- 32GB+ RAM recommended
- 500GB+ storage for full dataset

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/lanethefox/kaggle-rsna-aneurysm-detection.git
cd kaggle-rsna-aneurysm-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Create `requirements.txt`:
```txt
# Core dependencies
pydicom==2.4.0
numpy==1.24.3
polars==0.18.0
nibabel==5.1.0
scipy==1.10.1

# Deep learning (choose your framework)
torch==2.0.1
torchvision==0.15.2
# OR
# tensorflow==2.13.0

# ML utilities
scikit-learn==1.3.0
pandas==2.0.3
matplotlib==3.7.1
tqdm==4.65.0

# Medical imaging
SimpleITK==2.2.1
scikit-image==0.21.0

# Optional for advanced features
albumentations==1.3.1
monai==1.2.0
```

## Data Setup

### 1. Download Competition Data

```bash
# Install Kaggle API
pip install kaggle

# Set up Kaggle credentials (get from https://www.kaggle.com/account)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download competition data
kaggle competitions download -c rsna-intracranial-aneurysm-detection
unzip rsna-intracranial-aneurysm-detection.zip -d data/
```

### 2. Expected Directory Structure

```
kaggle-rsna-aneurysm-detection/
├── data/
│   ├── train.csv                 # Training labels
│   ├── train_localizers.csv      # Weak supervision annotations
│   ├── test.csv                  # Test metadata
│   ├── train/                    # Training DICOM files
│   │   └── {SeriesInstanceUID}/
│   │       └── {SOPInstanceUID}.dcm
│   └── test/                     # Test DICOM files
│       └── {SeriesInstanceUID}/
│           └── {SOPInstanceUID}.dcm
├── segmentations/                # Vessel segmentation masks
│   └── {SeriesInstanceUID}/
│       ├── {SeriesInstanceUID}.nii
│       └── {SeriesInstanceUID}_cowseg.nii
└── kaggle_evaluation/            # Evaluation API
```

### 3. Verify Data Integrity

```python
import pandas as pd
from pathlib import Path

# Check training data
train_df = pd.read_csv('data/train.csv')
print(f"Training samples: {len(train_df)}")
print(f"Unique series: {train_df['SeriesInstanceUID'].nunique()}")
print(f"Positive aneurysm cases: {train_df['Aneurysm Present'].sum()}")

# Check DICOM files
train_dir = Path('data/train')
series_dirs = list(train_dir.glob('*'))
print(f"Series directories: {len(series_dirs)}")
```

## Library Usage

### DICOM Loading Module

The `file.py` module provides utilities for loading DICOM series:

```python
from file import load_series_to_volume
from pathlib import Path

# Load a single series
series_path = Path("data/train/1.2.826.0.1.3680043.8.498.xxxxx")
volume, spacing, description = load_series_to_volume(series_path, to_hu=True)

print(f"Volume shape: {volume.shape}")  # (Z, H, W)
print(f"Voxel spacing (mm): {spacing}")  # (slice_thickness, row_spacing, col_spacing)
print(f"Series description: {description}")
```

### Advanced DICOM Processing

```python
import numpy as np
from pathlib import Path
import pydicom

def load_series_with_metadata(series_dir: Path):
    """Load series with additional metadata."""
    from file import load_series_to_volume
    
    # Basic loading
    volume, spacing, description = load_series_to_volume(series_dir)
    
    # Get additional metadata from first DICOM
    dcm_files = list(series_dir.glob("*.dcm"))
    if dcm_files:
        dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        metadata = {
            'PatientAge': getattr(dcm, 'PatientAge', None),
            'PatientSex': getattr(dcm, 'PatientSex', None),
            'Modality': getattr(dcm, 'Modality', None),
            'StudyDescription': getattr(dcm, 'StudyDescription', None),
            'SeriesInstanceUID': getattr(dcm, 'SeriesInstanceUID', None)
        }
    else:
        metadata = {}
    
    return volume, spacing, description, metadata

# Example usage
volume, spacing, desc, meta = load_series_with_metadata(series_path)
print(f"Modality: {meta.get('Modality')}")
print(f"Patient Age: {meta.get('PatientAge')}")
```

### Preprocessing Pipeline

```python
import numpy as np
from scipy import ndimage

class CTAPreprocessor:
    """Preprocessing for CTA scans."""
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0), 
                 window_center=100, window_width=700):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
    
    def resample_volume(self, volume, original_spacing):
        """Resample volume to target spacing."""
        zoom_factors = [
            original_spacing[i] / self.target_spacing[i] 
            for i in range(3)
        ]
        return ndimage.zoom(volume, zoom_factors, order=1)
    
    def apply_windowing(self, volume):
        """Apply CT windowing for vessel visualization."""
        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2
        volume = np.clip(volume, min_val, max_val)
        volume = (volume - min_val) / (max_val - min_val)
        return volume
    
    def preprocess(self, series_path):
        """Full preprocessing pipeline."""
        from file import load_series_to_volume
        
        # Load volume
        volume, spacing, _ = load_series_to_volume(series_path, to_hu=True)
        
        # Resample
        volume = self.resample_volume(volume, spacing)
        
        # Apply windowing
        volume = self.apply_windowing(volume)
        
        return volume, self.target_spacing

# Usage
preprocessor = CTAPreprocessor()
processed_volume, new_spacing = preprocessor.preprocess(series_path)
```

## Evaluation API

### Understanding the Gateway

The evaluation API processes one series at a time:

```python
from kaggle_evaluation.rsna_gateway import RSNAGateway, LABEL_COLS

# Initialize gateway
gateway = RSNAGateway(
    data_paths=('test.csv', 'test_series_dir'),
    file_share_dir='temp_dir'
)

# Label columns expected in submission
print("Expected labels:", LABEL_COLS)
print(f"Number of labels: {len(LABEL_COLS)}")
print(f"Primary label: {LABEL_COLS[-1]}")  # 'Aneurysm Present'
```

### Local Testing Setup

```python
from kaggle_evaluation.rsna_inference_server import RSNAInferenceServer
import numpy as np

class MyModel:
    """Your model implementation."""
    
    def predict(self, series_path):
        """Predict probabilities for a series."""
        # Load and preprocess
        from file import load_series_to_volume
        volume, spacing, _ = load_series_to_volume(series_path)
        
        # Your model inference here
        # For testing, return random probabilities
        probs = np.random.rand(14)
        
        return probs

# Test with evaluation server
server = RSNAInferenceServer()
model = MyModel()

# Simulate evaluation loop
def test_inference():
    test_series = Path("kaggle_evaluation/series")
    for series_dir in test_series.glob("*"):
        probs = model.predict(series_dir)
        print(f"Series: {series_dir.name}")
        print(f"Predictions: {probs}")
        break  # Test one series

test_inference()
```

## Model Development

### Dataset Class

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class RSNADataset(Dataset):
    """PyTorch dataset for RSNA competition."""
    
    def __init__(self, csv_path, series_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.series_dir = Path(series_dir)
        self.transform = transform
        self.label_cols = [col for col in self.df.columns 
                          if col not in ['SeriesInstanceUID', 'PatientAge', 
                                        'PatientSex', 'Modality']]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        
        # Load volume
        series_path = self.series_dir / series_id
        from file import load_series_to_volume
        volume, spacing, _ = load_series_to_volume(series_path)
        
        # Get labels
        labels = row[self.label_cols].values.astype(float)
        
        # Apply transforms
        if self.transform:
            volume = self.transform(volume)
        
        return {
            'volume': torch.FloatTensor(volume),
            'labels': torch.FloatTensor(labels),
            'series_id': series_id,
            'spacing': spacing
        }

# Usage
dataset = RSNADataset('data/train.csv', 'data/train')
print(f"Dataset size: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"Volume shape: {sample['volume'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
```

### Training Loop Template

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        volumes = batch['volume'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate model and compute AUC."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch['volume'].to(device)
            labels = batch['labels']
            
            outputs = model(volumes)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Compute per-label AUC
    aucs = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except:
            aucs.append(0.5)
    
    return aucs

# Training configuration
config = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# Initialize components
# model = YourModel()  # Define your model
# dataset = RSNADataset('data/train.csv', 'data/train')
# dataloader = DataLoader(dataset, batch_size=config['batch_size'])
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
```

## Submission Guide

### Creating Submission File

```python
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_submission(model, test_csv_path, test_series_dir, output_path):
    """Create submission file."""
    from kaggle_evaluation.rsna_gateway import LABEL_COLS, SUBMISSION_ID_COL
    
    # Read test metadata
    test_df = pd.read_csv(test_csv_path)
    
    # Initialize submission
    submission = pd.DataFrame()
    submission[SUBMISSION_ID_COL] = test_df[SUBMISSION_ID_COL].unique()
    
    # Generate predictions
    predictions = []
    for series_id in tqdm(submission[SUBMISSION_ID_COL]):
        series_path = Path(test_series_dir) / series_id
        
        # Get model predictions
        probs = model.predict(series_path)  # Should return 14 probabilities
        predictions.append(probs)
    
    # Add predictions to submission
    predictions = np.vstack(predictions)
    for i, col in enumerate(LABEL_COLS):
        submission[col] = predictions[:, i]
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Validate submission
    assert len(submission.columns) == 15  # ID + 14 labels
    assert submission[LABEL_COLS].min().min() >= 0
    assert submission[LABEL_COLS].max().max() <= 1
    
    return submission

# Usage
# submission = create_submission(model, 'test.csv', 'test', 'submission.csv')
```

### Validation Checklist

```python
def validate_submission(submission_path):
    """Validate submission format."""
    from kaggle_evaluation.rsna_gateway import LABEL_COLS, SUBMISSION_ID_COL
    
    df = pd.read_csv(submission_path)
    
    checks = {
        'Has ID column': SUBMISSION_ID_COL in df.columns,
        'Has all label columns': all(col in df.columns for col in LABEL_COLS),
        'Correct number of columns': len(df.columns) == 15,
        'Probabilities in [0,1]': (df[LABEL_COLS] >= 0).all().all() and 
                                  (df[LABEL_COLS] <= 1).all().all(),
        'No missing values': not df.isna().any().any(),
        'Correct column order': list(df.columns) == [SUBMISSION_ID_COL] + LABEL_COLS
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())

# validate_submission('submission.csv')
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

```python
# Solution: Process in smaller batches
def process_large_volume(volume, batch_size=32):
    """Process large volume in slices."""
    results = []
    for i in range(0, volume.shape[0], batch_size):
        batch = volume[i:i+batch_size]
        # Process batch
        result = your_processing_function(batch)
        results.append(result)
    return np.concatenate(results)
```

#### 2. DICOM Loading Errors

```python
def safe_dicom_load(series_path):
    """Robust DICOM loading with error handling."""
    try:
        from file import load_series_to_volume
        return load_series_to_volume(series_path)
    except Exception as e:
        print(f"Error loading {series_path}: {e}")
        # Return dummy data or handle appropriately
        return np.zeros((100, 512, 512)), (1.0, 1.0, 1.0), "error"
```

#### 3. Missing Modality Handling

```python
def handle_missing_modality(series_path):
    """Handle series with missing or unexpected modality."""
    dcm_files = list(Path(series_path).glob("*.dcm"))
    if not dcm_files:
        return None
    
    dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    modality = getattr(dcm, 'Modality', 'UNKNOWN')
    
    if modality == 'CT':
        # CTA processing
        return process_cta(series_path)
    elif modality == 'MR':
        # MRA/MRI processing
        return process_mra(series_path)
    else:
        # Default processing
        return process_default(series_path)
```

### Performance Optimization

```python
# 1. Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

# 2. Cache preprocessed data
import joblib

def cache_preprocessed(series_id, data, cache_dir='cache'):
    Path(cache_dir).mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"{series_id}.pkl"
    joblib.dump(data, cache_path)

def load_cached(series_id, cache_dir='cache'):
    cache_path = Path(cache_dir) / f"{series_id}.pkl"
    if cache_path.exists():
        return joblib.load(cache_path)
    return None

# 3. Parallel data loading
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)
```

## Additional Resources

- [Competition Page](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- [Discussion Forum](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion)
- [MONAI Documentation](https://docs.monai.io/) - Medical imaging framework
- [PyDICOM Documentation](https://pydicom.github.io/)
- [Competition Evaluation Metric Details](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview/evaluation)