import numpy as np
import pandas as pd
from scipy import sparse


CATEGORICAL_FIELDS = ("language_code", "publisher", "format")


def _text(value):
    return str(value).strip() if value is not None and not pd.isna(value) else ""


def _authors(value):
    if not isinstance(value, (tuple, list, set)):
        return []
    return sorted({_text(author) for author in value if _text(author)})


def build_features(books: pd.DataFrame, train_catalog, n_items=None):

    train_catalog = np.sort(np.unique(np.asarray(train_catalog, dtype=np.int64)))
    if not len(train_catalog):
        raise ValueError("Cannot fit metadata vocabulary without train items")
    n_items = int(n_items if n_items is not None else books.item_id.max() + 1)
    frame = books.set_index("item_id").reindex(np.arange(n_items))
    for field in (*CATEGORICAL_FIELDS, "authors", "num_pages", "publication_year"):
        if field not in frame:
            frame[field] = None
    names, vocabulary, numeric, offset = [], {}, {}, 0
    for field in (*CATEGORICAL_FIELDS, "authors"):
        if field == "authors":
            values = sorted({author for value in frame.loc[train_catalog, field] for author in _authors(value)})
        else:
            values = sorted({_text(value) for value in frame.loc[train_catalog, field] if _text(value)})

        vocabulary[field] = {"unknown_index": offset,
                             "values": {value: offset + i + 1 for i, value in enumerate(values)}}
        names.extend([field + "=<UNKNOWN>"] + [field + "=" + value for value in values])
        offset += len(values) + 1
    row_indices, col_indices, feature_values = [], [], []
    for field in (*CATEGORICAL_FIELDS, "authors"):
        fitted = vocabulary[field]
        for item, value in enumerate(frame[field]):
            tokens = _authors(value) if field == "authors" else [_text(value)]
            columns = sorted({fitted["values"].get(token, fitted["unknown_index"]) for token in tokens})
            if not columns:
                columns = [fitted["unknown_index"]]
            weight = 1 / np.sqrt(len(columns)) if field == "authors" else 1.0
            row_indices.extend([item] * len(columns))
            col_indices.extend(columns)
            feature_values.extend([weight] * len(columns))
    for field in ("num_pages", "publication_year"):
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        if field == "num_pages":
            values[(values <= 0) | (values > 100000)] = np.nan
            values = np.log1p(values)
        else:
            values[(values < 1000) | (values > 2100)] = np.nan
        fitted_values = values[train_catalog]
        valid = fitted_values[np.isfinite(fitted_values)]
        median = float(np.median(valid)) if len(valid) else 0.0
        mean = float(np.mean(valid)) if len(valid) else 0.0
        scale = float(np.std(valid)) if len(valid) else 1.0
        if scale < 1e-8:
            scale = 1.0
        missing = ~np.isfinite(values)
        standardized = (np.where(missing, median, values) - mean) / scale

        standardized = np.clip(standardized, -5.0, 5.0)
        numeric[field] = {"median": median, "mean": mean, "scale": scale,
                          "value_index": offset, "missing_index": offset + 1,
                          "transform": "log1p" if field == "num_pages" else "identity"}
        names.extend([field + ":scaled", field + ":missing"])
        for item in range(n_items):
            if standardized[item] != 0:
                row_indices.append(item); col_indices.append(offset); feature_values.append(standardized[item])
            if missing[item]:
                row_indices.append(item); col_indices.append(offset + 1); feature_values.append(1.0)
        offset += 2
    matrix = sparse.csr_matrix((np.asarray(feature_values, np.float32), (row_indices, col_indices)),
                               shape=(n_items, offset), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, {"feature_names": names, "vocabulary": vocabulary,
                    "numeric": numeric, "fitted_train_items": int(len(train_catalog)),
                    "n_features": int(offset), "allowed_fields": ["authors", *CATEGORICAL_FIELDS, "num_pages", "publication_year"]}
