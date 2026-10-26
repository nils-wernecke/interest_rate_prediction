from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
TARGET = "loan_int_rate"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, min_work_age: int = 14) -> pd.DataFrame:
    df = df.copy()
    df = df[df[TARGET].notnull()]
    df = df[df["person_emp_length"].notnull()]
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= df["person_age"] - min_work_age]
    
    return df.reset_index(drop=True)


def load_clean(path: Path = RAW_PATH, min_work_age: int = 14) -> pd.DataFrame:
    return clean(load_raw(path), min_work_age=min_work_age)