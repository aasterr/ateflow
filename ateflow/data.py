"""Getting user data into shape.

Three steps, each one explicit about what it did:
  - `read_csv`: bytes as they come out of real tools (Excel with ';' and
    decimal commas, Windows encodings, BOMs) into a DataFrame
  - `profile`: what each column looks like, so the user can see whether it
    fits the role they have in mind before estimating anything
  - `prepare`: the columns a question uses, coerced and checked; every
    dropped row and every recoding ends up in the report, never silently
"""

from __future__ import annotations

import io
import re

import pandas as pd

MAX_CATEGORIES = 50
MAX_STRATIFICATION_LEVELS = 10
MIN_ARM = 2  # below this there is nothing to compare
FEW_ARM = 10  # below this the estimate is fragile

TRUE_TOKENS = {"1", "true", "yes", "y", "si", "sì", "treated", "treatment", "on"}
FALSE_TOKENS = {"0", "false", "no", "n", "control", "untreated", "off"}
DELIMITERS = [",", ";", "\t", "|"]


class DataError(ValueError):
    """A problem with the user's data, phrased so the user can fix it."""


# ---------- reading ----------


def _decode(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        # Excel on Windows: cp1252 decodes any byte, so this cannot fail
        return raw.decode("cp1252", errors="replace"), "cp1252"


def _sniff_delimiter(text: str) -> str:
    """The delimiter that splits the first lines into the same number of fields."""
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    best, best_score = ",", (0.0, 0)
    for delim in DELIMITERS:
        counts = [ln.count(delim) for ln in lines]
        if not counts or counts[0] == 0:
            continue
        consistency = sum(c == counts[0] for c in counts) / len(counts)
        score = (consistency, counts[0])
        if score > best_score:
            best, best_score = delim, score
    return best


def _clean_name(name: object) -> str:
    # '#' starts a comment and '->' is an edge in the DAG text format
    return re.sub(r"\s+", " ", str(name).replace("->", "_").replace("#", "_")).strip()


def read_csv(raw: bytes) -> tuple[pd.DataFrame, dict]:
    """Reads a CSV whatever its dialect; returns the data and what was detected."""
    if not raw.strip():
        raise DataError("the file is empty")
    text, encoding = _decode(raw)
    delimiter = _sniff_delimiter(text)
    sample = "\n".join(text.splitlines()[1:200])
    decimal = "," if delimiter != "," and re.search(r"\d,\d", sample) else "."
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal,
                         skipinitialspace=True)
    except Exception as exc:
        raise DataError(f"could not read the file as CSV: {exc}") from exc

    renamed = {}
    names = []
    for col in df.columns:
        new = _clean_name(col)
        if not new or new.startswith("Unnamed:"):
            new = f"column_{len(names) + 1}"
        base, k = new, 2
        while new in names:
            new, k = f"{base}_{k}", k + 1
        if new != col:
            renamed[str(col)] = new
        names.append(new)
    df.columns = names

    if df.shape[1] < 2:
        raise DataError(
            f"found a single column ({names[0]!r}): the file does not look delimited "
            "by comma, semicolon, tab or pipe"
        )
    if len(df) == 0:
        raise DataError("the file has a header but no data rows")
    delimiter_name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[delimiter]
    return df, {
        "rows": len(df),
        "encoding": encoding,
        "delimiter": delimiter_name,
        "decimal": "comma" if decimal == "," else "point",
        "renamed": renamed,
    }


# ---------- profiling ----------


def column_kind(series: pd.Series) -> str:
    """binary | discrete | continuous | categorical | identifier | text | constant | empty."""
    values = series.dropna()
    unique = values.nunique()
    if unique == 0:
        return "empty"
    if unique == 1:
        return "constant"
    if unique == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(values):
        if (pd.api.types.is_integer_dtype(values) and unique == len(values)
                and len(values) > 20 and values.is_monotonic_increasing):
            return "identifier"
        return "discrete" if unique <= MAX_STRATIFICATION_LEVELS else "continuous"
    if len(values) > 20 and unique > 0.9 * len(values):
        return "identifier"
    return "categorical" if unique <= MAX_CATEGORIES else "text"


def profile(df: pd.DataFrame) -> list[dict]:
    """One entry per column: kind, missing values, a few example values."""
    out = []
    for name in df.columns:
        s = df[name]
        examples = [str(v) for v in pd.unique(s.dropna())[:4]]
        out.append({
            "name": name,
            "kind": column_kind(s),
            "numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "unique": int(s.nunique()),
            "missing": int(s.isna().sum()),
            "examples": examples,
        })
    return out


# ---------- preparing a question ----------


def _token(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
        if number in (0.0, 1.0):
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def _encode_binary(series: pd.Series, name: str, role: str, chosen: str | None) -> tuple[pd.Series, dict]:
    """Maps a two-valued column to 0/1. Returns the codes and which value became 1."""
    values = list(pd.unique(series))
    if len(values) == 1:
        raise DataError(
            f"{name!r} takes a single value ({values[0]}) in the rows used: "
            "there is nothing to compare"
        )
    if len(values) > 2:
        shown = ", ".join(str(v) for v in values[:5])
        raise DataError(
            f"the {role} {name!r} must have exactly two values, it has {len(values)} "
            f"({shown}{', …' if len(values) > 5 else ''}). Recode it into two groups first."
        )
    labels = {str(v): v for v in values}
    if chosen is not None:
        if str(chosen) not in labels:
            raise DataError(f"{chosen!r} is not a value of {name!r}: use one of {sorted(labels)}")
        positive = labels[str(chosen)]
    else:
        tokens = {_token(v): v for v in values}
        if set(tokens) <= TRUE_TOKENS | FALSE_TOKENS and len(set(tokens) & TRUE_TOKENS) == 1:
            positive = tokens[(set(tokens) & TRUE_TOKENS).pop()]
        else:
            a, b = sorted(labels)
            raise DataError(
                f"{name!r} has the values {a!r} and {b!r}: choose which one counts as "
                f"{'treated' if role == 'treatment' else 'the positive outcome'}"
            )
    negative = next(v for v in values if str(v) != str(positive))
    codes = (series.astype(str) == str(positive)).astype(float)
    return codes, {"1": str(positive), "0": str(negative)}


def prepare(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    method: str = "g-computation",
    treated_value: str | None = None,
    outcome_positive: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """The rows and columns a question needs, coerced to what the estimators expect.

    Rows with a missing value in any used column are dropped (complete cases)
    and counted per column. Everything that cannot be fixed safely raises a
    DataError that says what to change.
    """
    used = list(dict.fromkeys([treatment, outcome, *adjustment_set]))
    warnings: list[str] = []

    missing = {c: int(df[c].isna().sum()) for c in used if df[c].isna().any()}
    work = df.loc[df[used].notna().all(axis=1), used].copy()
    if len(work) == 0:
        raise DataError("no row has all the variables of this question filled in")
    dropped = len(df) - len(work)
    if dropped:
        share = dropped / len(df)
        detail = ", ".join(f"{c}: {n}" for c, n in missing.items())
        warnings.append(f"{dropped} of {len(df)} rows dropped for missing values ({detail})")
        if share > 0.3:
            warnings.append(
                f"{share:.0%} of the rows are incomplete: if they are not missing at random "
                "the estimate is biased"
            )

    work[treatment], treatment_coding = _encode_binary(
        work[treatment], treatment, "treatment", treated_value)

    outcome_coding = None
    if pd.api.types.is_bool_dtype(work[outcome]) or (
        not pd.api.types.is_numeric_dtype(work[outcome]) and work[outcome].nunique() <= 2
    ):
        work[outcome], outcome_coding = _encode_binary(
            work[outcome], outcome, "outcome", outcome_positive)
    elif not pd.api.types.is_numeric_dtype(work[outcome]):
        numeric = pd.to_numeric(work[outcome], errors="coerce")
        bad = work.loc[numeric.isna(), outcome].iloc[0]
        raise DataError(
            f"the outcome {outcome!r} must be numeric or two-valued; {bad!r} is neither "
            "(units, thousands separators or text labels must be removed first)"
        )
    else:
        work[outcome] = work[outcome].astype(float)

    for c in adjustment_set:
        kind = column_kind(work[c])
        if kind == "identifier":
            raise DataError(
                f"{c!r} looks like an identifier ({work[c].nunique()} distinct values in "
                f"{len(work)} rows): adjusting for it would give every row its own group. "
                "Remove its arrows from the DAG"
            )
        if kind == "text":
            raise DataError(
                f"{c!r} has {work[c].nunique()} distinct text values: too many categories to "
                f"adjust for (the limit is {MAX_CATEGORIES}). Group them first"
            )
        if kind == "constant":
            warnings.append(f"{c!r} never varies in the rows used: adjusting for it changes nothing")
        if method == "stratification" and kind == "continuous":
            raise DataError(
                f"stratification needs discrete confounders, and {c!r} has "
                f"{work[c].nunique()} distinct values. Use g-computation, IPW or AIPW, "
                "or bin it into a few groups"
            )

    arms = work[treatment].value_counts()
    n_treated, n_control = int(arms.get(1.0, 0)), int(arms.get(0.0, 0))
    if min(n_treated, n_control) < MIN_ARM:
        raise DataError(
            f"only {n_treated} treated and {n_control} control rows: at least "
            f"{MIN_ARM} of each are needed"
        )
    if min(n_treated, n_control) < FEW_ARM:
        warnings.append(
            f"only {n_treated} treated and {n_control} control rows: the estimate is fragile"
        )

    return work, {
        "rows_in": len(df),
        "rows_used": len(work),
        "missing_by_column": missing,
        "treated": n_treated,
        "control": n_control,
        "treatment_coding": treatment_coding,
        "outcome_coding": outcome_coding,
        "warnings": warnings,
    }
