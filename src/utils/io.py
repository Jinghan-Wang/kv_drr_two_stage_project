from typing import Any, Tuple
import numpy as np
import nibabel as nib


def load_nifti_2d(path: str) -> Tuple[np.ndarray, np.ndarray, Any]:
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # squeeze to 2D if possible
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D NIfTI (or squeezeable to 2D), got shape: {data.shape} from {path}")

    return data, img.affine, img.header


def save_nifti_2d(data: np.ndarray, affine: np.ndarray, header: Any, path: str) -> None:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array to save, got {data.shape}")
    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    nib.save(img, path)
