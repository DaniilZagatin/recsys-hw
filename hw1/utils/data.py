from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd

from .splitting import make_temporal_split


SOURCE_PAGE = "https://mengtingwan.github.io/data/goodreads.html"
SOURCE_BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"
REVIEWS_FILE = "goodreads_reviews_comics_graphic.json.gz"
BOOKS_FILE = "goodreads_books_comics_graphic.json.gz"
DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
STATIC_FIELDS = ("title", "authors", "language_code", "publisher", "format",
                 "num_pages", "publication_year")


@dataclass
class PreparedDataset:
    ratings: pd.DataFrame
    books: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    profile: dict

    @property
    def n_users(self):
        return len(self.user_ids)

    @property
    def n_items(self):
        return len(self.item_ids)


Dataset = PreparedDataset


def download_file(url, destination):

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "RecSys-homework/1.0"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary.replace(destination)
    return destination


def _book_id(value):
    value = str(value or "").strip()
    return str(int(value)) if value.isdigit() and int(value) > 0 else None


def _user_id(value):
    value = str(value or "").strip()
    return value if value and value != "0" else None


def _parse_dates(series):


    return pd.to_datetime(series.replace("", None), format=DATE_FORMAT,
                          errors="coerce", utc=True)


def clean_reviews(frame: pd.DataFrame):

    frame = frame.copy()
    for field in ("user_id", "book_id", "review_id", "rating", "date_added", "date_updated", "read_at", "started_at"):
        if field not in frame:
            frame[field] = ""
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["source_user_id"] = frame.user_id.map(_user_id)
    frame["source_item_id"] = frame.book_id.map(_book_id)
    rating = pd.to_numeric(frame.rating, errors="coerce")
    dates = {key: _parse_dates(frame[key].fillna("").astype(str))
             for key in ("date_added", "date_updated", "read_at", "started_at")}
    date_profile = {}
    for key, parsed in dates.items():
        nonempty = frame[key].fillna("").astype(str).str.strip().ne("")
        date_profile[key] = {
            "empty": int((~nonempty).sum()), "invalid_nonempty": int((nonempty & parsed.isna()).sum()),
            "parsed": int(parsed.notna().sum()),
            "minimum_utc": parsed.min().isoformat() if parsed.notna().any() else None,
            "maximum_utc": parsed.max().isoformat() if parsed.notna().any() else None,
        }
    anomalies = {
        "updated_before_added": int((dates["date_updated"] < dates["date_added"]).sum()),
        "read_after_updated": int((dates["read_at"] > dates["date_updated"]).sum()),
        "read_before_added": int((dates["read_at"] < dates["date_added"]).sum()),
        "started_after_read": int((dates["started_at"] > dates["read_at"]).sum()),
    }
    valid_ids = frame.source_user_id.notna() & frame.source_item_id.notna()
    valid_rating = rating.between(1, 5) & rating.eq(np.floor(rating))
    valid_time = dates["date_updated"].notna()
    stats = {
        "raw_rows": int(len(frame)), "invalid_id_rows": int((~valid_ids).sum()),
        "raw_users_with_valid_ids": int(frame.source_user_id.nunique()),
        "raw_items_with_valid_ids": int(frame.source_item_id.nunique()),
        "raw_rating_distribution": {str(k): int(v) for k, v in rating.value_counts(dropna=False).sort_index().items()},
        "rating_zero_missing_rows": int(rating.eq(0).sum()),
        "invalid_or_missing_rating_rows_including_zero": int((~valid_rating).sum()),
        "invalid_updated_timestamp_rows": int((~valid_time).sum()),
        "date_fields": date_profile, "date_anomalies_reported_not_filtered": anomalies,
    }
    frame["rating"] = rating
    frame["timestamp"] = dates["date_updated"].astype("int64") // 10**9
    clean = frame.loc[valid_ids & valid_rating & valid_time].copy()
    stats["rows_before_pair_deduplication"] = int(len(clean))


    clean = clean.sort_values(["timestamp", "_source_row"], kind="stable")
    clean = clean.drop_duplicates(["source_user_id", "source_item_id"], keep="last")
    stats["duplicate_pair_rows_removed"] = stats["rows_before_pair_deduplication"] - len(clean)
    clean["rating"] = clean.rating.astype(np.float32)
    return clean[["source_user_id", "source_item_id", "rating", "timestamp"]].reset_index(drop=True), stats


def _read_review_fields(path):
    fields = ("user_id", "book_id", "review_id", "rating", "date_added", "date_updated", "read_at", "started_at")
    rows, bad = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                rows.append({key: row.get(key, "") for key in fields})
            except (json.JSONDecodeError, AttributeError):
                bad += 1
    return pd.DataFrame(rows), bad


def _read_book_fields(path):
    rows, bad, invalid_ids, duplicates = {}, 0, 0, 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                item = _book_id(row.get("book_id"))
                if item is None:
                    invalid_ids += 1
                    continue
                duplicates += int(item in rows)
                static = {key: row.get(key, "") for key in STATIC_FIELDS}
                static["authors"] = sorted({str(a["author_id"]) for a in (row.get("authors") or [])
                                             if isinstance(a, dict) and a.get("author_id")})
                rows[item] = static
            except (json.JSONDecodeError, AttributeError):
                bad += 1
    return rows, {"metadata_rows": len(rows), "invalid_json_rows": bad,
                  "invalid_id_rows": invalid_ids, "duplicate_book_rows": duplicates}


def _file_info(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"filename": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest(), "source_url": SOURCE_BASE + Path(path).name}


def prepare_dataset(data_dir, *, download=True, profile_path=None) -> PreparedDataset:

    data_dir = Path(data_dir)
    paths = {name: data_dir / name for name in (REVIEWS_FILE, BOOKS_FILE)}
    for name, path in paths.items():
        if not path.exists():
            if not download:
                raise FileNotFoundError(path)
            download_file(SOURCE_BASE + name, path)
    raw, invalid_json = _read_review_fields(paths[REVIEWS_FILE])
    clean, review_stats = clean_reviews(raw)
    del raw
    if clean.empty:
        raise ValueError("No valid rated review observations remain after cleaning")
    user_ids = np.asarray(sorted(clean.source_user_id.unique()), dtype=str)
    item_ids = np.asarray(sorted(clean.source_item_id.unique(), key=int), dtype=str)
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    ratings = pd.DataFrame({
        "user_id": clean.source_user_id.map(user_map).to_numpy(dtype=np.int64),
        "item_id": clean.source_item_id.map(item_map).to_numpy(dtype=np.int64),
        "rating": clean.rating.to_numpy(dtype=np.float32),
        "timestamp": clean.timestamp.to_numpy(dtype=np.int64),
    }).sort_values(["timestamp", "user_id", "item_id"], kind="stable").reset_index(drop=True)
    metadata, book_stats = _read_book_fields(paths[BOOKS_FILE])
    books = pd.DataFrame([{"item_id": index, "source_item_id": str(item),
                           **metadata.get(str(item), {key: [] if key == "authors" else "" for key in STATIC_FIELDS})}
                          for index, item in enumerate(item_ids)])
    book_stats["rated_items_without_metadata"] = sum(str(item) not in metadata for item in item_ids)
    for field in STATIC_FIELDS:
        book_stats["rated_items_missing_" + field] = int(sum(not bool(value) for value in books[field]))
    profile = {
        "source_page": SOURCE_PAGE, "files": [_file_info(path) for path in paths.values()],
        "observation": "Available review version at date_updated; not original rating time or read_at.",
        "cleaning": "Rating 0 is missing; require valid IDs, integer rating 1..5, parsed date_updated; keep latest user/book version.",
        "forbidden_filters_not_applied": ["sampling", "activity", "popularity", "calendar_window", "minimum_user_history", "minimum_item_support"],
        "raw_invalid_json_review_rows": invalid_json, "reviews": review_stats, "books": book_stats,
        "clean": {"rows": int(len(ratings)), "users": len(user_ids), "items": len(item_ids),
                  "rating_distribution": {str(int(k)): int(v) for k, v in ratings.rating.value_counts().sort_index().items()},
                  "minimum_observation_utc": pd.Timestamp(int(ratings.timestamp.min()), unit="s", tz="UTC").isoformat(),
                  "maximum_observation_utc": pd.Timestamp(int(ratings.timestamp.max()), unit="s", tz="UTC").isoformat(),
                  "density": float(len(ratings) / (len(user_ids) * len(item_ids)))},
    }
    profile["split"] = make_temporal_split(ratings).summary()
    if profile_path is None:
        profile_path = Path(__file__).resolve().parents[1] / "results" / "data_profile.json"
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PreparedDataset(ratings, books, user_ids, item_ids, profile)
