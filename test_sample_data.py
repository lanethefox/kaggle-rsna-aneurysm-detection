#!/usr/bin/env python3
"""
Test script to verify sample data is working correctly.
Run this to ensure the sample dataset is properly set up.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def test_sample_data():
    """Run tests on sample data."""
    
    print("Testing Sample Data Setup")
    print("=" * 50)
    
    sample_dir = Path("sample_data")
    
    # Test 1: Check directory structure
    print("\n1. Checking directory structure...")
    required_dirs = ["train", "test", "segmentations"]
    required_files = ["train.csv", "test.csv", "train_localizers.csv", "metadata.json"]
    
    all_good = True
    for dir_name in required_dirs:
        if (sample_dir / dir_name).exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ✗ {dir_name}/ missing")
            all_good = False
    
    for file_name in required_files:
        if (sample_dir / file_name).exists():
            print(f"  ✓ {file_name} exists")
        else:
            print(f"  ✗ {file_name} missing")
            all_good = False
    
    if not all_good:
        print("\n❌ Some files are missing. Run create_sample_data.py first.")
        return False
    
    # Test 2: Load and validate CSVs
    print("\n2. Loading CSV files...")
    
    train_df = pd.read_csv(sample_dir / "train.csv")
    test_df = pd.read_csv(sample_dir / "test.csv")
    localizers_df = pd.read_csv(sample_dir / "train_localizers.csv")
    
    print(f"  ✓ train.csv: {len(train_df)} samples")
    print(f"  ✓ test.csv: {len(test_df)} samples")
    print(f"  ✓ train_localizers.csv: {len(localizers_df)} annotations")
    
    # Test 3: Check DICOM loading
    print("\n3. Testing DICOM loading...")
    
    try:
        from file import load_series_to_volume
        
        # Try loading test series
        test_series = list((sample_dir / "test").glob("*"))
        if test_series:
            series_path = test_series[0]
            if series_path.is_dir():
                volume, spacing, description = load_series_to_volume(series_path)
                print(f"  ✓ Loaded series: {series_path.name[:20]}...")
                print(f"    - Shape: {volume.shape}")
                print(f"    - Spacing: {spacing}")
                print(f"    - Description: {description}")
        else:
            print("  ⚠ No test series found with DICOM files")
    except ImportError:
        print("  ✗ Could not import load_series_to_volume from file.py")
    except Exception as e:
        print(f"  ✗ Error loading DICOM: {e}")
    
    # Test 4: Check segmentations
    print("\n4. Checking segmentations...")
    
    seg_dirs = list((sample_dir / "segmentations").glob("*"))
    if seg_dirs:
        print(f"  ✓ Found {len(seg_dirs)} segmentation directories")
        for seg_dir in seg_dirs[:2]:  # Check first 2
            nii_files = list(seg_dir.glob("*.nii"))
            if nii_files:
                print(f"    - {seg_dir.name[:20]}...: {len(nii_files)} NIfTI files")
    else:
        print("  ⚠ No segmentation directories found")
    
    # Test 5: Validate labels
    print("\n5. Validating label columns...")
    
    expected_labels = [
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
        'Aneurysm Present'
    ]
    
    missing_labels = [label for label in expected_labels if label not in train_df.columns]
    if missing_labels:
        print(f"  ✗ Missing labels: {missing_labels}")
    else:
        print(f"  ✓ All 14 label columns present")
        
        # Check label values
        label_values = train_df[expected_labels].values
        if np.all((label_values == 0) | (label_values == 1)):
            print(f"  ✓ All labels are binary (0 or 1)")
        else:
            print(f"  ✗ Non-binary label values found")
    
    # Test 6: Summary statistics
    print("\n6. Dataset Statistics:")
    print(f"  - Modalities: {train_df['Modality'].value_counts().to_dict()}")
    print(f"  - Aneurysm Present: {train_df['Aneurysm Present'].sum()}/{len(train_df)}")
    print(f"  - Patient Sex: {train_df['PatientSex'].value_counts().to_dict()}")
    
    # Test 7: Memory footprint
    print("\n7. Memory footprint:")
    total_size = sum(f.stat().st_size for f in sample_dir.rglob('*') if f.is_file())
    print(f"  Total sample data size: {total_size / (1024**2):.2f} MB")
    
    print("\n" + "=" * 50)
    print("✅ Sample data validation complete!")
    print("\nYou can now use the sample data for development.")
    print("Example: python3 -c \"from file import load_series_to_volume; print('Import successful')\"")
    
    return True

if __name__ == "__main__":
    success = test_sample_data()
    sys.exit(0 if success else 1)