#!/usr/bin/env python3
"""
Create a small sample dataset for development and testing.
Samples a few series from the full dataset to create a lightweight version.
"""

import pandas as pd
import shutil
import random
from pathlib import Path
import json
import numpy as np

def create_sample_dataset(
    num_train_samples=5,
    num_test_samples=2,
    num_segmentation_samples=2,
    random_seed=42
):
    """Create a sample dataset with a small subset of data."""
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create directories
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "train").mkdir(exist_ok=True)
    (sample_dir / "test").mkdir(exist_ok=True)
    (sample_dir / "segmentations").mkdir(exist_ok=True)
    
    print("Creating sample dataset...")
    
    # 1. Sample training data
    print(f"\n1. Sampling {num_train_samples} training series...")
    
    # Read full train CSV
    train_df = pd.read_csv("train.csv")
    
    # Sample diverse cases: mix of positive and negative, different modalities
    positive_cases = train_df[train_df["Aneurysm Present"] == 1]
    negative_cases = train_df[train_df["Aneurysm Present"] == 0]
    
    # Sample proportionally
    num_positive = min(2, len(positive_cases), num_train_samples // 2)
    num_negative = num_train_samples - num_positive
    
    sampled_positive = positive_cases.sample(n=num_positive, random_state=random_seed)
    sampled_negative = negative_cases.sample(n=num_negative, random_state=random_seed)
    
    sample_train_df = pd.concat([sampled_positive, sampled_negative]).reset_index(drop=True)
    
    # Save sampled train CSV
    sample_train_df.to_csv(sample_dir / "train.csv", index=False)
    print(f"  Saved {len(sample_train_df)} samples to sample_data/train.csv")
    
    # Copy corresponding DICOM series (if available)
    train_series_copied = []
    for series_id in sample_train_df["SeriesInstanceUID"]:
        # Check kaggle_evaluation/series for available series
        source_path = Path("kaggle_evaluation/series") / series_id
        if source_path.exists():
            dest_path = sample_dir / "train" / series_id
            if not dest_path.exists():
                print(f"  Copying series {series_id[:20]}...")
                shutil.copytree(source_path, dest_path)
                train_series_copied.append(series_id)
        else:
            print(f"  Series {series_id[:20]}... not found in kaggle_evaluation/series")
    
    # 2. Sample test data
    print(f"\n2. Sampling {num_test_samples} test series...")
    
    # Get available test series
    test_series_dir = Path("kaggle_evaluation/series")
    available_test_series = [d.name for d in test_series_dir.iterdir() if d.is_dir()]
    
    if available_test_series:
        sampled_test_series = random.sample(
            available_test_series, 
            min(num_test_samples, len(available_test_series))
        )
        
        # Create test CSV
        test_df = pd.DataFrame({
            "SeriesInstanceUID": sampled_test_series
        })
        test_df.to_csv(sample_dir / "test.csv", index=False)
        print(f"  Saved {len(test_df)} test samples to sample_data/test.csv")
        
        # Copy test series
        for series_id in sampled_test_series:
            source_path = test_series_dir / series_id
            dest_path = sample_dir / "test" / series_id
            if not dest_path.exists():
                print(f"  Copying test series {series_id[:20]}...")
                shutil.copytree(source_path, dest_path)
    
    # 3. Sample segmentations
    print(f"\n3. Sampling {num_segmentation_samples} segmentation files...")
    
    seg_dir = Path("segmentations")
    if seg_dir.exists():
        available_segs = [d.name for d in seg_dir.iterdir() if d.is_dir()]
        
        # Prefer segmentations that match our sampled training data
        matching_segs = [s for s in available_segs if s in sample_train_df["SeriesInstanceUID"].values]
        
        if matching_segs:
            sampled_segs = matching_segs[:num_segmentation_samples]
        else:
            sampled_segs = random.sample(
                available_segs,
                min(num_segmentation_samples, len(available_segs))
            )
        
        for seg_id in sampled_segs:
            source_path = seg_dir / seg_id
            dest_path = sample_dir / "segmentations" / seg_id
            if not dest_path.exists():
                print(f"  Copying segmentation {seg_id[:20]}...")
                shutil.copytree(source_path, dest_path)
    
    # 4. Sample localizers
    print("\n4. Creating sample localizers...")
    
    if Path("train_localizers.csv").exists():
        localizers_df = pd.read_csv("train_localizers.csv")
        
        # Filter to match sampled training series
        sample_localizers = localizers_df[
            localizers_df["SeriesInstanceUID"].isin(sample_train_df["SeriesInstanceUID"])
        ]
        
        sample_localizers.to_csv(sample_dir / "train_localizers.csv", index=False)
        print(f"  Saved {len(sample_localizers)} localizer annotations")
    
    # 5. Create metadata file
    print("\n5. Creating metadata...")
    
    metadata = {
        "description": "Sample dataset for RSNA Intracranial Aneurysm Detection",
        "num_train_series": len(sample_train_df),
        "num_test_series": len(sampled_test_series) if 'sampled_test_series' in locals() else 0,
        "num_segmentations": len(sampled_segs) if 'sampled_segs' in locals() else 0,
        "train_series_copied": train_series_copied,
        "modalities": sample_train_df["Modality"].value_counts().to_dict(),
        "aneurysm_present_ratio": {
            "positive": int(sample_train_df["Aneurysm Present"].sum()),
            "negative": int((~sample_train_df["Aneurysm Present"].astype(bool)).sum())
        },
        "random_seed": random_seed
    }
    
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("Sample dataset created successfully!")
    print(f"Location: {sample_dir.absolute()}")
    print(f"Total size should be much smaller than original dataset")
    print("\nDataset contents:")
    print(f"  - {metadata['num_train_series']} training series")
    print(f"  - {metadata['num_test_series']} test series")
    print(f"  - {metadata['num_segmentations']} segmentations")
    print(f"  - Modalities: {metadata['modalities']}")
    print(f"  - Aneurysm cases: {metadata['aneurysm_present_ratio']}")
    
    return metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset")
    parser.add_argument("--train-samples", type=int, default=5, 
                       help="Number of training samples")
    parser.add_argument("--test-samples", type=int, default=2,
                       help="Number of test samples")
    parser.add_argument("--seg-samples", type=int, default=2,
                       help="Number of segmentation samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_sample_dataset(
        num_train_samples=args.train_samples,
        num_test_samples=args.test_samples,
        num_segmentation_samples=args.seg_samples,
        random_seed=args.seed
    )