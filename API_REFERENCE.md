# API Reference - RSNA Intracranial Aneurysm Detection

## Core Modules

### `file.py` - DICOM Loading

#### `load_series_to_volume(series_dir, to_hu=True)`

Loads a DICOM series and converts it to a numpy volume.

**Parameters:**
- `series_dir` (Path): Directory containing DICOM files for a series
- `to_hu` (bool): Convert CT values to Hounsfield Units (default: True)

**Returns:**
- `volume` (np.ndarray): 3D array of shape (Z, H, W) with int16 dtype
- `spacing` (tuple): Voxel spacing in mm as (slice_thickness, row_spacing, col_spacing)
- `description` (str): Series description from DICOM metadata

**Example:**
```python
from pathlib import Path
from file import load_series_to_volume

series_path = Path("data/train/1.2.826.0.1.3680043.8.498.xxxxx")
volume, spacing, desc = load_series_to_volume(series_path)
```

**Notes:**
- Automatically sorts slices by InstanceNumber
- Handles missing SliceThickness by inferring from ImagePositionPatient
- Applies RescaleSlope and RescaleIntercept for CT/CTA modalities

---

### `kaggle_evaluation.rsna_gateway` - Evaluation Gateway

#### Class: `RSNAGateway`

Gateway for iterating through test data in the competition format.

**Attributes:**
- `LABEL_COLS` (list): 14 anatomical label column names
- `SUBMISSION_ID_COL` (str): Series identifier column name

**Methods:**

##### `__init__(data_paths=None, file_share_dir=None)`
Initialize the gateway with data paths.

**Parameters:**
- `data_paths` (tuple): (test_csv_path, test_dicom_dir) paths
- `file_share_dir` (str): Directory for temporary file sharing

##### `generate_data_batches()`
Generator that yields test data batches.

**Yields:**
- `batch` (tuple): ((series_files,), series_uid) for each test series

**Example:**
```python
from kaggle_evaluation.rsna_gateway import RSNAGateway

gateway = RSNAGateway(
    data_paths=('test.csv', 'test_series'),
    file_share_dir='temp'
)

for batch in gateway.generate_data_batches():
    series_files, series_uid = batch
    # Process series
```

---

### `kaggle_evaluation.rsna_inference_server` - Inference Server

#### Class: `RSNAInferenceServer`

Main server class for competition submission.

**Methods:**

##### `_get_gateway_for_test(data_paths=None, file_share_dir=None)`
Returns configured RSNAGateway instance for testing.

**Example:**
```python
from kaggle_evaluation.rsna_inference_server import RSNAInferenceServer

server = RSNAInferenceServer()
gateway = server._get_gateway_for_test(
    data_paths=('test.csv', 'test_series')
)
```

---

## Constants and Configuration

### Label Columns

The 14 anatomical labels in order:

```python
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',  # Weight: 13 in evaluation
]
```

### Submission Format

```python
# Required DataFrame structure
submission_df = pd.DataFrame({
    'SeriesInstanceUID': series_ids,  # Unique series identifiers
    **{col: probabilities for col in LABEL_COLS}  # Probabilities [0,1]
})
```

---

## Utility Functions

### Preprocessing Utilities

```python
def normalize_hu_values(volume, window_center, window_width):
    """Apply CT windowing to Hounsfield Units."""
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    volume = np.clip(volume, min_val, max_val)
    return (volume - min_val) / (max_val - min_val)

def resample_volume(volume, original_spacing, target_spacing):
    """Resample volume to target voxel spacing."""
    from scipy import ndimage
    zoom_factors = [orig/target for orig, target in 
                   zip(original_spacing, target_spacing)]
    return ndimage.zoom(volume, zoom_factors, order=1)
```

### Validation Utilities

```python
def validate_predictions(predictions, series_ids):
    """Validate prediction format."""
    assert predictions.shape == (len(series_ids), 14)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
    return True

def compute_weighted_auc(y_true, y_pred, weights=None):
    """Compute weighted multi-label AUC."""
    from sklearn.metrics import roc_auc_score
    
    if weights is None:
        weights = [1] * 13 + [13]  # Default competition weights
    
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            auc = 0.5
        aucs.append(auc * weights[i])
    
    return sum(aucs) / sum(weights)
```

---

## Data Structures

### Training DataFrame Schema

```python
# train.csv structure
train_df = pd.DataFrame({
    'SeriesInstanceUID': str,     # Unique series identifier
    'PatientAge': str,             # Age in years (e.g., '045Y')
    'PatientSex': str,             # 'M' or 'F'
    'Modality': str,               # 'CTA', 'MRA', etc.
    # ... 14 binary label columns (0 or 1)
})
```

### Localizers DataFrame Schema

```python
# train_localizers.csv structure
localizers_df = pd.DataFrame({
    'SeriesInstanceUID': str,     # Series identifier
    'SOPInstanceUID': str,         # Specific slice identifier
    'x': float,                    # X coordinate of aneurysm
    'y': float,                    # Y coordinate of aneurysm
    'Location': str,               # Text description of location
})
```

---

## Error Handling

### DICOM Loading Errors

```python
class DICOMLoadError(Exception):
    """Raised when DICOM series cannot be loaded."""
    pass

def safe_load_series(series_dir):
    """Load series with comprehensive error handling."""
    try:
        return load_series_to_volume(series_dir)
    except FileNotFoundError:
        raise DICOMLoadError(f"Series not found: {series_dir}")
    except Exception as e:
        raise DICOMLoadError(f"Failed to load {series_dir}: {str(e)}")
```

### Submission Validation Errors

```python
class SubmissionError(Exception):
    """Raised when submission format is invalid."""
    pass

def validate_submission_format(df):
    """Validate submission DataFrame format."""
    required_cols = ['SeriesInstanceUID'] + LABEL_COLS
    
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise SubmissionError(f"Missing columns: {missing}")
    
    if df[LABEL_COLS].isna().any().any():
        raise SubmissionError("Submission contains NaN values")
    
    if not ((df[LABEL_COLS] >= 0) & (df[LABEL_COLS] <= 1)).all().all():
        raise SubmissionError("Probabilities must be in [0, 1]")
    
    return True
```

---

## Performance Considerations

### Memory Management

```python
# Process large volumes in chunks
def process_in_chunks(volume, chunk_size=64):
    """Process volume in memory-efficient chunks."""
    for i in range(0, volume.shape[0], chunk_size):
        chunk = volume[i:i+chunk_size]
        yield process_chunk(chunk)

# Clear memory after processing
import gc
def clear_memory():
    """Force garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def parallel_preprocess(series_paths, num_workers=4):
    """Preprocess multiple series in parallel."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_series_to_volume, series_paths))
    return results
```

---

## Environment Variables

```bash
# Optional configuration via environment variables
export RSNA_DATA_DIR=/path/to/data          # Override data directory
export RSNA_CACHE_DIR=/path/to/cache        # Cache preprocessed data
export RSNA_NUM_WORKERS=4                   # Parallel processing workers
export RSNA_BATCH_SIZE=8                    # Default batch size
export RSNA_DEBUG=1                         # Enable debug logging
```

---

## Type Hints

```python
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import numpy as np

def load_series_to_volume(
    series_dir: Union[str, Path],
    to_hu: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float], str]:
    """Type-annotated version of load_series_to_volume."""
    ...

def create_submission(
    predictions: np.ndarray,
    series_ids: List[str],
    label_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create submission DataFrame with type hints."""
    ...
```