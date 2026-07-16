"""Audit DICOM series in cases not completed by the YOLO dataset pipeline."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from finetune.config.settings import settings


PIPELINE_VERSION = "yolo-dataset-v4-unicode-png"
REPORT_DIR = settings.database_path / "reports"
CASE_REPORT = REPORT_DIR / "unprocessed_cases_series_audit.csv"
SERIES_REPORT = REPORT_DIR / "unprocessed_dicom_series.csv"

SPECIFIC_TAGS = [
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "ImageOrientationPatient",
    "ImageType",
    "Modality",
    "EchoTime",
    "RepetitionTime",
    "InversionTime",
    "SliceThickness",
]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value)
    return str(value)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_axial(orientation: Any) -> bool:
    try:
        rounded = np.round(np.asarray(orientation, dtype=float)).astype(int)
        return rounded.size >= 6 and np.array_equal(
            rounded[:6], np.asarray([1, 0, 0, 0, 1, 0])
        )
    except (TypeError, ValueError):
        return False


def _is_current(marker: Path) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == PIPELINE_VERSION
    except (FileNotFoundError, OSError, UnicodeError):
        return False


@dataclass
class SeriesSummary:
    uid: str
    files: int = 0
    series_number: str = ""
    modality: str = ""
    description: str = ""
    protocol: str = ""
    sequence_name: str = ""
    scanning_sequence: str = ""
    image_type: str = ""
    axial: bool = False
    echo_time: float | None = None
    echo_times: set[float] = field(default_factory=set)
    repetition_time: float | None = None
    inversion_time: float | None = None
    slice_thickness: float | None = None
    current_filter_match: bool = False
    evidence: set[str] = field(default_factory=set)

    def add(self, dataset: pydicom.dataset.Dataset) -> None:
        self.files += 1
        self.series_number = self.series_number or _text(
            getattr(dataset, "SeriesNumber", "")
        )
        self.modality = self.modality or _text(getattr(dataset, "Modality", ""))
        self.description = self.description or _text(
            getattr(dataset, "SeriesDescription", "")
        )
        self.protocol = self.protocol or _text(getattr(dataset, "ProtocolName", ""))
        self.sequence_name = self.sequence_name or _text(
            getattr(dataset, "SequenceName", "")
        )
        self.scanning_sequence = self.scanning_sequence or _text(
            getattr(dataset, "ScanningSequence", "")
        )
        self.image_type = self.image_type or _text(
            getattr(dataset, "ImageType", "")
        )
        current_echo_time = _number(getattr(dataset, "EchoTime", None))
        current_repetition_time = _number(getattr(dataset, "RepetitionTime", None))
        if self.echo_time is None:
            self.echo_time = current_echo_time
        if current_echo_time is not None:
            self.echo_times.add(current_echo_time)
        if self.repetition_time is None:
            self.repetition_time = current_repetition_time
        self.inversion_time = self.inversion_time or _number(
            getattr(dataset, "InversionTime", None)
        )
        self.slice_thickness = self.slice_thickness or _number(
            getattr(dataset, "SliceThickness", None)
        )

        orientation = getattr(dataset, "ImageOrientationPatient", None)
        axial = _is_axial(orientation)
        self.axial = self.axial or axial

        description_lower = self.description.casefold()
        combined = " ".join(
            [
                self.description,
                self.protocol,
                self.sequence_name,
                self.scanning_sequence,
            ]
        ).casefold()
        scanning_upper = self.scanning_sequence.upper().split("\\")

        explicit_flair = any(
            token in combined
            for token in ("flair", "dark fluid", "fluid attenuated", "tirm")
        )
        explicit_t2 = "t2" in combined
        inversion_recovery = "IR" in scanning_upper
        parameter_candidate = (
            current_echo_time is not None
            and current_echo_time >= 70
            and current_repetition_time is not None
            and current_repetition_time >= 2000
        )

        if explicit_flair:
            self.evidence.add("explicit_flair")
        if explicit_t2:
            self.evidence.add("explicit_t2")
        if inversion_recovery:
            self.evidence.add("inversion_recovery")
        if parameter_candidate:
            self.evidence.add("long_te_tr")

        mode_ok = (
            "flair" in description_lower
            or "fluid" in description_lower
            or inversion_recovery
        )
        self.current_filter_match = self.current_filter_match or (
            mode_ok and axial and "t2" in description_lower
        )

    @property
    def candidate(self) -> bool:
        return bool(
            self.evidence
            & {"explicit_flair", "explicit_t2", "inversion_recovery", "long_te_tr"}
        )

    def rejection_reasons(self) -> str:
        reasons: list[str] = []
        description_lower = self.description.casefold()
        scanning_upper = self.scanning_sequence.upper().split("\\")
        if "t2" not in description_lower:
            reasons.append("SeriesDescription has no T2")
        if not (
            "flair" in description_lower
            or "fluid" in description_lower
            or "IR" in scanning_upper
        ):
            reasons.append("no FLAIR/fluid/IR marker")
        if not self.axial:
            reasons.append("not exact axial orientation")
        return "; ".join(reasons)


def _iter_candidate_files(case_path: Path) -> Iterable[Path]:
    for path in case_path.rglob("*"):
        if not path.is_file():
            continue
        if any(parent.name.casefold() == "markups" for parent in path.parents):
            continue
        yield path


def _read_case(case_path: Path) -> tuple[dict[str, SeriesSummary], int]:
    series: dict[str, SeriesSummary] = {}
    unreadable = 0
    for path in _iter_candidate_files(case_path):
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=SPECIFIC_TAGS,
            )
        except (InvalidDicomError, OSError, ValueError):
            unreadable += 1
            continue

        uid = _text(getattr(dataset, "SeriesInstanceUID", ""))
        if not uid:
            unreadable += 1
            continue
        summary = series.setdefault(uid, SeriesSummary(uid=uid))
        summary.add(dataset)
    return series, unreadable


def _case_classification(series: Iterable[SeriesSummary]) -> str:
    values = list(series)
    if not values:
        return "NO_READABLE_DICOM"
    if any(item.current_filter_match for item in values):
        return "HAS_CURRENT_FILTER_MATCH"
    if any("explicit_flair" in item.evidence for item in values):
        return "FLAIR_REJECTED_BY_FILTER"
    if any("explicit_t2" in item.evidence for item in values):
        return "T2_REJECTED_BY_FILTER"
    if any(
        item.evidence & {"inversion_recovery", "long_te_tr"} for item in values
    ):
        return "POSSIBLE_T2_FLAIR_BY_PARAMETERS"
    return "NO_T2_FLAIR_EVIDENCE"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    case_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()

    cases = [
        case
        for case in settings.input_data_path.iterdir()
        if case.is_dir()
        and not _is_current(settings.processed_data_path / case.name / ".complete")
    ]

    for index, case in enumerate(sorted(cases, key=lambda item: item.name), start=1):
        series, unreadable = _read_case(case)
        classification = _case_classification(series.values())
        classifications[classification] += 1
        candidates = [item for item in series.values() if item.candidate]

        case_rows.append(
            {
                "case": case.name,
                "classification": classification,
                "readable_series": len(series),
                "candidate_t2_flair_series": len(candidates),
                "current_filter_matches": sum(
                    item.current_filter_match for item in series.values()
                ),
                "unreadable_or_non_dicom_files": unreadable,
                "candidate_descriptions": " | ".join(
                    sorted(
                        {
                            item.description or item.protocol or "(no description)"
                            for item in candidates
                        }
                    )
                ),
            }
        )

        for item in sorted(
            series.values(), key=lambda value: (value.series_number, value.description)
        ):
            series_rows.append(
                {
                    "case": case.name,
                    "series_uid": item.uid,
                    "series_number": item.series_number,
                    "files": item.files,
                    "modality": item.modality,
                    "description": item.description,
                    "protocol": item.protocol,
                    "sequence_name": item.sequence_name,
                    "scanning_sequence": item.scanning_sequence,
                    "image_type": item.image_type,
                    "axial": item.axial,
                    "echo_time": item.echo_time,
                    "all_echo_times": "|".join(
                        str(value) for value in sorted(item.echo_times)
                    ),
                    "repetition_time": item.repetition_time,
                    "inversion_time": item.inversion_time,
                    "slice_thickness": item.slice_thickness,
                    "evidence": "|".join(sorted(item.evidence)),
                    "current_filter_match": item.current_filter_match,
                    "current_filter_rejection": item.rejection_reasons(),
                }
            )

        print(f"Audited {index}/{len(cases)} cases", end="\r", flush=True)

    _write_csv(CASE_REPORT, case_rows)
    _write_csv(SERIES_REPORT, series_rows)
    print(" " * 60, end="\r")
    print(f"Audited cases: {len(cases)}")
    for classification, count in sorted(classifications.items()):
        print(f"{classification}: {count}")
    print(f"Case report: {CASE_REPORT}")
    print(f"Series report: {SERIES_REPORT}")


if __name__ == "__main__":
    main()
