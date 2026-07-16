"""Integrity and metadata smoke checks for the committed demo study."""
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "fileStorage" / "processed_data" / "689"

NIFTI_FILES = (
    DEMO / "T1.nii",
    DEMO / "T2.nii",
    DEMO / "Predict" / "segmentation.nii",
    DEMO / "ParcelationMask" / "parcelation.nii",
)


def test_demo_nifti_files_are_present_and_have_clean_headers():
    for path in NIFTI_FILES:
        header = path.read_bytes()[:352]
        assert len(header) == 352
        assert struct.unpack("<I", header[:4])[0] == 348
        # NIfTI-1 text fields that commonly retain source descriptions.
        assert not header[148:228].rstrip(b"\0")
        assert not header[228:252].rstrip(b"\0")


def test_demo_support_files_and_license_are_present():
    expected = (
        ROOT / "fileStorage" / "for_volume_calc" / "689" / "mri" / "T1.mgz",
        ROOT
        / "fileStorage"
        / "for_volume_calc"
        / "689"
        / "mri"
        / "wmparc.DKTatlas.mapped.mgz",
        ROOT / "fileStorage" / "fast_surfer_lic" / "license.txt",
    )
    for path in expected:
        assert path.is_file()
        assert path.stat().st_size > 0
