import hashlib, json, datetime as dt
from pathlib import Path
import pandas as pd, yaml, yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def flat_tickers(t):
    out = [t["index"], t["vix"]]
    out += list(t["stocks"].values())
    return out

def validate(df, name):
    issues = []
    if df.index.duplicated().any():
        issues.append("duplicate dates")
    if (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        issues.append("non-positive prices")
    if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
        issues.append("High < max(Open, Close)")
    if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
        issues.append("Low > min(Open, Close)")
    if df.isna().any().any():
        issues.append(f"NaNs: {int(df.isna().sum().sum())}")
    return issues

manifest = {}
for t in flat_tickers(cfg["tickers"]):
    df = yf.download(t, start=cfg["start_date"], end=cfg["end_date"],
                     auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)   # flatten MultiIndex
    df = df.sort_index()
    df.index.name = "Date"

    path = RAW / f"{t.replace('^', '')}.csv"
    df.to_csv(path)

    manifest[t] = {
        "file": path.name,
        "rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "issues": validate(df, t),
    }
    print(t, manifest[t]["rows"], manifest[t]["first_date"],
          manifest[t]["last_date"], manifest[t]["issues"])

manifest["_downloaded_on"] = str(dt.date.today())
(RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))