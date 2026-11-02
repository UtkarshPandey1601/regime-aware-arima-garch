import pandas as pd, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
FINAL = PROC / "final"
FINAL.mkdir(exist_ok=True)

rd = lambda f: pd.read_csv(f, index_col="Date", parse_dates=True).sort_index()
index_name = cfg["tickers"]["index"]
stocks = list(cfg["tickers"]["stocks"].values())
names = [index_name] + stocks

nifty = rd(RAW / "NSEI.csv")
vix = rd(RAW / "INDIAVIX.csv")
cal = nifty.index
flag = {}

def add(d, why):
    if d is not None:
        flag.setdefault(d, []).append(why)

def next_day(d):
    pos = cal.searchsorted(d)             # first calendar date >= d
    return cal[pos] if pos < len(cal) else None

# 1. Real sessions missing from the Nifty file (VIX has them): next return spans 2 sessions
for d in vix.index.difference(cal):
    add(next_day(d), f"2-session return (Nifty file lacks {d.date()})")

# 2. Stale stock rows (flat or zero volume): that day and the next are unreliable
for t in stocks:
    df = rd(RAW / f"{t}.csv")
    df = df.loc[cal.intersection(df.index)]
    for d in df.index[(df.High == df.Low) | (df.Volume == 0)]:
        add(d, f"stale row in {t}")
        pos = cal.get_loc(d)
        if pos + 1 < len(cal):
            add(cal[pos + 1], f"day after stale row in {t}")

print(f"Removing {len(flag)} dates from ALL series:")
for d in sorted(flag):
    print(" ", d.date(), "|", "; ".join(flag[d]))

for t in names:
    fn = t.replace("^", "") + ".csv"
    df = rd(PROC / fn)
    out = df.drop(index=[d for d in flag if d in df.index])
    out.to_csv(FINAL / fn)
    print(t, len(df), "->", len(out))