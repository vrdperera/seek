"""Canonical project paths used by CLIs and the application."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

RETRIEVAL_DATA_DIR = PROCESSED_DATA_DIR / "retrieval"
DEFAULT_MANIFEST = RETRIEVAL_DATA_DIR / "manifest.csv"
DEFAULT_DETECTION_DATA = INTERIM_DATA_DIR / "detection" / "deepfashion2.yaml"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
RETRIEVAL_CHECKPOINT = CHECKPOINTS_DIR / "retrieval" / "best.pt"
DETECTOR_CHECKPOINT = CHECKPOINTS_DIR / "detector" / "best.pt"
CATALOGUE_INDEX = ARTIFACTS_DIR / "indexes" / "catalogue.pt"
DETECTOR_RUNS_DIR = ARTIFACTS_DIR / "runs" / "detector"
ULTRALYTICS_CONFIG_DIR = ARTIFACTS_DIR / "cache" / "ultralytics"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
