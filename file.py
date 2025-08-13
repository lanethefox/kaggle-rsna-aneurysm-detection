import pydicom, numpy as np
from pathlib import Path

def load_series_to_volume(series_dir: Path, to_hu=True):
    # 1) read & sort by InstanceNumber
    dcm_paths = sorted(series_dir.glob("*.dcm"),
                       key=lambda p: int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber))
    dcms = [pydicom.dcmread(p) for p in dcm_paths]

    # 2) stack to (Z, H, W)
    vol = np.stack([d.pixel_array for d in dcms]).astype(np.int16)

    # 3) get spacing (mm)
    px_spacing = tuple(map(float, dcms[0].PixelSpacing))     # (row, col)
    try:
        slice_thickness = float(dcms[0].SliceThickness)
    except Exception:
        # Fallback: infer from ImagePositionPatient z-diffs
        zs = [float(getattr(d, "ImagePositionPatient", [0,0,i])[2]) for i, d in enumerate(dcms)]
        slice_thickness = np.median(np.diff(sorted(zs)))
    spacing = (slice_thickness, px_spacing[0], px_spacing[1])

    # 4) CT/CTA → HU
    if to_hu and hasattr(dcms[0], "RescaleIntercept"):
        intercept = float(dcms[0].RescaleIntercept)
        slope     = float(dcms[0].RescaleSlope)
        vol = (vol * slope + intercept).astype(np.int16)

    return vol, spacing, dcms[0].SeriesDescription if hasattr(dcms[0],"SeriesDescription") else "unknown"
