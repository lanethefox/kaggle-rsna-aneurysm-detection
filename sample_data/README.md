# Sample Data for RSNA Intracranial Aneurysm Detection

This directory contains a small sample dataset for development and testing purposes. It includes a subset of the full competition data to allow developers to work without downloading the entire 250GB+ dataset.

## Contents

- **train/** - Sample training DICOM series (2 series, ~30MB each)
- **test/** - Sample test DICOM series (2 series, ~30MB each)  
- **segmentations/** - Sample vessel segmentation NIfTI files (2 series)
- **train.csv** - Training labels for sample series
- **train_localizers.csv** - Weak localization annotations
- **test.csv** - Test series metadata
- **metadata.json** - Information about the sample dataset

## Dataset Statistics

- Total size: ~69MB (vs 250GB+ full dataset)
- Training samples: 3 series (1 positive, 2 negative for aneurysm)
- Test samples: 2 series
- Modalities: CTA (2), MRI T2 (1)
- Includes examples of both aneurysm-positive and negative cases

## Quick Start

```python
from pathlib import Path
import pandas as pd
from file import load_series_to_volume

# Load sample training data
train_df = pd.read_csv('sample_data/train.csv')
print(f"Sample training series: {len(train_df)}")

# Load a sample DICOM series
series_id = train_df.iloc[0]['SeriesInstanceUID']
series_path = Path(f'sample_data/test/{series_id}')  # Using test dir as it has DICOM files

if series_path.exists():
    volume, spacing, description = load_series_to_volume(series_path)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")
```

## Using Sample Data for Development

### 1. Testing DICOM Loading

```python
from pathlib import Path
from file import load_series_to_volume

# Test with sample series
for series_dir in Path('sample_data/test').glob('*'):
    if series_dir.is_dir():
        volume, spacing, desc = load_series_to_volume(series_dir)
        print(f"Loaded {series_dir.name}: shape={volume.shape}")
```

### 2. Testing Preprocessing Pipeline

```python
import numpy as np
from scipy import ndimage

def test_preprocessing():
    # Load sample volume
    series_dirs = list(Path('sample_data/test').glob('*'))
    if series_dirs:
        volume, spacing, _ = load_series_to_volume(series_dirs[0])
        
        # Test windowing
        windowed = np.clip(volume, -100, 400)
        
        # Test resampling
        target_spacing = (1.0, 1.0, 1.0)
        zoom = [s/t for s, t in zip(spacing, target_spacing)]
        resampled = ndimage.zoom(volume, zoom, order=1)
        
        print(f"Original: {volume.shape}, Resampled: {resampled.shape}")
```

### 3. Testing Model Inference

```python
class DummyModel:
    def predict(self, series_path):
        # Mock prediction for testing
        return np.random.rand(14)  # 14 label probabilities

# Test with sample data
model = DummyModel()
test_df = pd.read_csv('sample_data/test.csv')

for series_id in test_df['SeriesInstanceUID']:
    series_path = Path(f'sample_data/test/{series_id}')
    if series_path.exists():
        probs = model.predict(series_path)
        print(f"Predictions for {series_id[:20]}...: {probs[:3]}...")
```

## Regenerating Sample Data

To create a new sample dataset with different parameters:

```bash
python3 create_sample_data.py --train-samples 5 --test-samples 3 --seg-samples 3 --seed 123
```

## Note on Full Dataset

This sample data is for development only. For actual training and submission:
1. Download the full dataset from Kaggle
2. Place it in the appropriate directories (excluded from git)
3. Update paths in your code to use the full data

## File Structure Comparison

```
Full Dataset (~250GB):              Sample Dataset (~69MB):
├── train/                          ├── train/
│   └── 3000+ series                │   └── 0 series (metadata only)
├── test/                           ├── test/
│   └── 100+ series                 │   └── 2 series
├── segmentations/                  ├── segmentations/
│   └── 500+ series                 │   └── 2 series
├── train.csv (3000+ rows)          ├── train.csv (3 rows)
├── train_localizers.csv            ├── train_localizers.csv (1 row)
└── test.csv (100+ rows)            └── test.csv (2 rows)
```