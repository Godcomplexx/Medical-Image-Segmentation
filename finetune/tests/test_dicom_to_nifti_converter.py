import logging
from pathlib import Path

import pytest

from finetune.src.filetools import dicom_to_nifti_converter as converter


def _prepare_source(tmp_path: Path) -> Path:
    source = tmp_path / "исследование" / "DICOM"
    source.mkdir(parents=True)
    (source / "срез").write_bytes(b"dicom-one")
    nested = source / "вложенная"
    nested.mkdir()
    (nested / "срез").write_bytes(b"dicom-two")
    return source


def test_conversion_stages_unicode_input_under_ascii_unique_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _prepare_source(tmp_path)
    ascii_temp = tmp_path / "ascii-temp"
    output = tmp_path / "output"
    monkeypatch.setenv("FINETUNE_DICOM_TEMP", str(ascii_temp))
    monkeypatch.setattr(converter.pydicom, "dcmread", lambda _: object())

    staged_names: list[str] = []

    def fake_convert_directory(
        dicom_directory: str,
        output_folder: str,
        compression: bool,
        reorient: bool,
    ) -> None:
        assert str(Path(dicom_directory)).isascii()
        staged_names.extend(
            path.name for path in Path(dicom_directory).iterdir() if path.is_file()
        )
        (Path(output_folder) / "series.nii.gz").write_bytes(b"nifti")

    monkeypatch.setattr(
        converter.dicom2nifti, "convert_directory", fake_convert_directory
    )

    converter.process_dicom_to_nifti(
        source,
        tmp_path / "результат" / "DICOM_TEMP",
        output,
    )

    assert staged_names == ["00000000.dcm", "00000001.dcm"]
    assert (output / "series.nii.gz").is_file()
    assert not any(ascii_temp.iterdir())


def test_conversion_rejects_partial_dicom2nifti_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _prepare_source(tmp_path)
    monkeypatch.setenv("FINETUNE_DICOM_TEMP", str(tmp_path / "ascii-temp"))
    monkeypatch.setattr(converter.pydicom, "dcmread", lambda _: object())

    def fake_convert_directory(
        dicom_directory: str,
        output_folder: str,
        compression: bool,
        reorient: bool,
    ) -> None:
        (Path(output_folder) / "partial.nii.gz").write_bytes(b"partial")
        logging.getLogger("dicom2nifti.convert_dir").warning(
            "Unable to read: 00000001.dcm"
        )

    monkeypatch.setattr(
        converter.dicom2nifti, "convert_directory", fake_convert_directory
    )

    with pytest.raises(converter.DicomConversionError, match="partial conversion"):
        converter.process_dicom_to_nifti(
            source,
            tmp_path / "result" / "DICOM_TEMP",
            tmp_path / "output",
        )


def test_conversion_rejects_missing_nifti_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _prepare_source(tmp_path)
    monkeypatch.setenv("FINETUNE_DICOM_TEMP", str(tmp_path / "ascii-temp"))
    monkeypatch.setattr(converter.pydicom, "dcmread", lambda _: object())
    monkeypatch.setattr(
        converter.dicom2nifti, "convert_directory", lambda *args, **kwargs: None
    )

    with pytest.raises(converter.DicomConversionError, match="produced no NIfTI"):
        converter.process_dicom_to_nifti(
            source,
            tmp_path / "result" / "DICOM_TEMP",
            tmp_path / "output",
        )
