from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.database import init_db, save_dataframe

BASE_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_DIR = BASE_DIR / "data" / "synthetic"
SYNTHETIC_CURRENT_DIR = SYNTHETIC_DATA_DIR / "current"
SYNTHETIC_SNAPSHOTS_DIR = SYNTHETIC_DATA_DIR / "snapshots"
SYNTHETIC_ARCHIVE_DIR = SYNTHETIC_DATA_DIR / "archive"
SYNTHETIC_API_BASE_URL = os.getenv("SYNTHETIC_API_BASE_URL", "http://fastapi:8000")

RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
)

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_STRATEGY))
SESSION.mount("http://", HTTPAdapter(max_retries=RETRY_STRATEGY))

TABLE_FILE_MAP = {
    "students": "studentInfo.csv",
    "assessments": "studentAssessment.csv",
    "vle": "studentVle.csv",
}

DEDUP_KEYS = {
    "students": ["code_module", "code_presentation", "id_student"],
    "assessments": ["id_assessment", "id_student", "date_submitted"],
    "vle": ["code_module", "code_presentation", "id_student", "id_site", "date"],
}


def fetch_synthetic_batch(
    target_date: str | None = None,
    batch_size: int = 32,
    timeout: int = 120,
) -> dict[str, list[dict[str, object]]]:
    params = {"batch_size": batch_size}
    if target_date:
        params["target_date"] = target_date

    response = SESSION.get(
        f"{SYNTHETIC_API_BASE_URL}/synthetic-data",
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["data"]


def _append_table(table_name: str, records: list[dict[str, object]]) -> int:
    frame = pd.DataFrame(records)
    if frame.empty:
        return 0

    SYNTHETIC_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    destination = SYNTHETIC_CURRENT_DIR / TABLE_FILE_MAP[table_name]

    if destination.exists():
        existing = pd.read_csv(destination)
        frame = pd.concat([existing, frame], ignore_index=True)

    frame = frame.drop_duplicates(subset=DEDUP_KEYS[table_name], keep="last")
    frame.to_csv(destination, index=False)
    return int(len(frame))


def _write_snapshot(table_name: str, records: list[dict[str, object]], target_date: str) -> None:
    snapshot_dir = SYNTHETIC_SNAPSHOTS_DIR / target_date
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(snapshot_dir / TABLE_FILE_MAP[table_name], index=False)


def collect_synthetic_data(target_date: str, batch_size: int = 64) -> dict[str, object]:
    payload = fetch_synthetic_batch(target_date=target_date, batch_size=batch_size)
    init_db()
    totals: dict[str, int] = {}

    for table_name, records in payload.items():
        totals[table_name] = _append_table(table_name, records)
        _write_snapshot(table_name, records, target_date)
        save_dataframe(
            table_name=f"synthetic_{table_name}",
            df=pd.DataFrame(records),
            if_exists="replace",
        )

    return {
        "target_date": target_date,
        "batch_size": batch_size,
        "tables": totals,
        "current_dir": str(SYNTHETIC_CURRENT_DIR),
    }


def archive_and_reset_synthetic_data(run_label: str | None = None) -> str | None:
    if not SYNTHETIC_CURRENT_DIR.exists():
        return None

    csv_files = list(SYNTHETIC_CURRENT_DIR.glob("*.csv"))
    if not csv_files:
        return None

    archive_label = run_label or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    archive_dir = SYNTHETIC_ARCHIVE_DIR / archive_label
    archive_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        shutil.copy2(csv_path, archive_dir / csv_path.name)
        csv_path.unlink()

    return str(archive_dir)
