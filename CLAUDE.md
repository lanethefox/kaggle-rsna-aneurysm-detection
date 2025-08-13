# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for RSNA Intracranial Aneurysm Detection. The goal is to classify aneurysm presence and anatomical locations in brain imaging data (CTA/MRA/MRI) using multilabel classification with weighted AUC ROC evaluation.

## Key Files and Structure

- `train.csv`: Training metadata with SeriesInstanceUID, patient info, and 14 binary labels
- `train_localizers.csv`: Per-aneurysm weak localization annotations  
- `segmentations/`: NIfTI vessel segmentation masks for subset of training data
- `kaggle_evaluation/`: Kaggle evaluation API implementation
  - `rsna_gateway.py`: Handles test data iteration and submission format
  - `rsna_inference_server.py`: Main inference server class
  - `series/`: Test DICOM files organized by SeriesInstanceUID
- `file.py`: DICOM loading utility for converting series to numpy volumes

## Commands for Development

### Running Tests
```bash
# Test DICOM loading
python file.py

# Test evaluation gateway locally
python -m kaggle_evaluation.rsna_inference_server
```

### Data Exploration
```bash
# Count training samples
wc -l train.csv

# Check segmentation files
ls segmentations/ | wc -l

# View DICOM structure
ls kaggle_evaluation/series/*/
```

## Architecture and Approach

### Data Processing Pipeline
1. **DICOM Loading**: Use `file.py:load_series_to_volume()` to convert DICOM series to numpy arrays with proper HU conversion and spacing extraction
2. **Modality Handling**: Support CTA/MRA/MRI with modality-specific normalization
3. **Preprocessing**: Resampling to consistent voxel spacing, optional skull stripping, vessel-focused ROI extraction

### Model Architecture
- 3D CNN classifiers for volumetric data or 2.5D hybrid approaches
- Multilabel binary classification with 14 outputs (Aneurysm Present + 13 anatomical sites)
- Optional auxiliary localization head using train_localizers.csv weak supervision
- Vessel-aware features using segmentation masks where available

### Evaluation Strategy
- Patient-level stratified k-fold cross-validation
- Weighted AUC ROC with heavy weight on "Aneurysm Present" (weight=13)
- Ensemble across folds for final predictions

### Submission Format
The submission must follow exact schema with 14 probability columns:
- SeriesInstanceUID (identifier)
- 14 binary probability columns in exact order as in train.csv

## Important Constraints

1. **Kaggle Runtime**: No internet access, strict CPU/GPU time limits
2. **Test Data**: Process one series at a time via evaluation API
3. **DICOM Tags**: Only whitelisted tags available at test time
4. **Deterministic**: Fixed seeds for reproducibility
5. **Memory**: Efficient batching and gradient checkpointing required

## Label Definitions

The 14 binary labels (in order):
1. Left Infraclinoid Internal Carotid Artery
2. Right Infraclinoid Internal Carotid Artery  
3. Left Supraclinoid Internal Carotid Artery
4. Right Supraclinoid Internal Carotid Artery
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation
14. **Aneurysm Present** (weight=13 in evaluation)

## Development Tips

- Use mixed precision training (AMP) to fit within memory constraints
- Implement modality dropout for robustness to missing sequences
- Cache preprocessed volumes to speed up training
- Monitor per-label AUCs during cross-validation
- Test submission format compliance before final submission