import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RAW, PROC, RES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"
PROC.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
manifest = json.loads((RAW / "manifest.json").read_text())

def load(ticker):
    f = RAW / manifest[ticker]["file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != manifest[ticker]["sha256"]:
        print(f"WARNING {ticker}: checksum differs from manifest "
              "(line-endings change after clone is the usual cause)")
    return pd.read_csv(f, index_col="Date", parse_dates=True).sort_index()

def range_proxies(df):
    # OHLC are unadjusted, Adj Close is adjusted: scale OHLC by the same factor
    k = df["Adj Close"] / df["Close"]
    o, h, l, c = (df[x] * k for x in ["Open", "High", "Low", "Close"])
    hl, co = np.log(h / l), np.log(c / o)
    park = (hl ** 2) / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    return park * 1e4, gk * 1e4          # in (%)^2, same units as returns^2

index_name = cfg["tickers"]["index"]
vix_name = cfg["tickers"]["vix"]
series = [index_name] + list(cfg["tickers"]["stocks"].values())

nsei = load(index_name)
calendar = nsei.index                     # master trading calendar
top_rows = []

for t in series:
    df = load(t)
    extra = df.index.difference(calendar)
    missing = calendar.difference(df.index)
    print(f"\n=== {t} ===")
    print(f"rows in file: {len(df)} | not in Nifty calendar: {len(extra)} | "
          f"Nifty days missing here: {len(missing)}")
    if len(extra):
        print("  extra dates:", [str(d.date()) for d in extra][:25])
    if len(missing):
        print("  missing dates:", [str(d.date()) for d in missing][:25])
    df = df.loc[calendar.intersection(df.index)]

    flat = (df["High"] == df["Low"]).sum()
    zero_vol = (df["Volume"] == 0).sum() if t != index_name else "n/a"
    print(f"  flat days (High==Low): {flat} | zero-volume days: {zero_vol}")

    out = pd.DataFrame(index=df.index)
    out["adj_close"] = df["Adj Close"]
    out["ret"] = 100 * np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    out["ret_sq"] = out["ret"] ** 2
    out["parkinson"], out["gk"] = range_proxies(df)
    out = out.dropna(subset=["ret"])      # drops the first day
    out.to_csv(PROC / f"{t.replace('^', '')}.csv")

    print(f"  usable returns: {len(out)} | {out.index.min().date()} to {out.index.max().date()}")
    print(f"  GK negative days: {(out['gk'] < 0).sum()}")
    big = out["ret"].abs().nlargest(10)
    for d, v in big.items():
        top_rows.append({"series": t, "date": d.date(), "ret": out.loc[d, "ret"]})

pd.DataFrame(top_rows).to_csv(RES / "top10_abs_returns.csv", index=False)
print("\nTop 10 absolute returns per series saved to results/top10_abs_returns.csv")
print(pd.DataFrame(top_rows).to_string(index=False))

# India VIX: level series on the same calendar (used for regime cross-checks)
vix = load(vix_name).reindex(calendar)
print(f"\nVIX missing on {vix['Close'].isna().sum()} Nifty days")
vix[["Close"]].rename(columns={"Close": "vix"}).to_csv(PROC / "INDIAVIX.csv")