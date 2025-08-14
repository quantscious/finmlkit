"""
Unified Binance trades -> H5 pipeline (download + process + klines) for Spot and Perpetual markets.

Usage example:
python scripts/binance2h5.py \
  --market spot \
  --tickers BTCUSDT ETHUSDT \
  --start 2021-01 \
  --end now \
  --workdir /Users/you/PROJECTS/QTS/data \
  --workers 4 \
  --overwrite-klines 1

Outputs files named SYMBOL_MARKET_YYMM-YYMM.h5 under {workdir}/h5, where MARKET is one of:
- SPOT
- PERPum (USDⓈ-M futures)
- PERPcm (COIN-M futures)

Raw monthly trade ZIPs (+ optional .CHECKSUM) are stored under {workdir}/raw/{market_path}/trades/{symbol}
where market_path is one of: spot, um, cm.

Notes:
- If an existing H5 for the same symbol+market matches the same start but an earlier end, it will be
  renamed to the new full-range name before appending new months (extension).
- Requires pandas, requests, tqdm, and finmlkit installed in the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import multiprocessing
import os
import re
import zipfile
from dataclasses import dataclass
from hashlib import md5, sha256
from multiprocessing import Pool
from threading import Thread, Lock
from typing import List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from finmlkit.bar.data_model import TradesData
from finmlkit.bar.io import AddTimeBarH5

BINANCE_BASE = "https://data.binance.vision"


# ------------------------- Date helpers -------------------------
@dataclass
class Month:
    year: int
    month: int

    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def yymm(self) -> str:
        return f"{self.year % 100:02d}{self.month:02d}"


def parse_ym(s: str) -> Month:
    s = s.strip().lower()
    if s == "now":
        today = dt.date.today()
        return Month(today.year, today.month)
    m = re.match(r"^(\d{4})(?:[-./]?(\d{1,2}))?$", s)
    if not m:
        raise ValueError(f"Invalid date format: {s}. Use YYYY or YYYY-MM or 'now'.")
    year = int(m.group(1))
    mon = int(m.group(2)) if m.group(2) else 1
    if mon < 1 or mon > 12:
        raise ValueError(f"Invalid month in: {s}")
    return Month(year, mon)


def month_range(start: Month, end: Month) -> List[Month]:
    if (end.year, end.month) < (start.year, start.month):
        raise ValueError("end must be >= start")
    out: List[Month] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(Month(y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


# ------------------------- Checksum & CSV utils (inlined) -------------------------
HEX32 = re.compile(r"\b[a-fA-F0-9]{32}\b")
HEX64 = re.compile(r"\b[a-fA-F0-9]{64}\b")


def _read_checksum_file(checksum_path: str) -> Optional[str]:
    if not os.path.exists(checksum_path):
        return None
    try:
        with open(checksum_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        cands64 = HEX64.findall(content)
        cands32 = HEX32.findall(content)
        if cands64:
            return cands64[0].lower()
        if cands32:
            return cands32[0].lower()
        return None
    except Exception:
        return None


def _file_hash(path: str, algo: str) -> Optional[str]:
    try:
        h = sha256() if algo == "sha256" else md5()
    except Exception:
        return None
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def verify_zip_checksum(zip_path: str, checksum_path: str) -> bool:
    digest = _read_checksum_file(checksum_path)
    if digest is None:
        print(f"[warn] CHECKSUM not found or unreadable for {os.path.basename(zip_path)}; skipping verification.")
        return True
    for algo in ("sha256", "md5"):
        comp = _file_hash(zip_path, algo)
        if comp and comp.lower() == digest:
            return True
    print(f"[error] Checksum mismatch for {os.path.basename(zip_path)}")
    return False


def load_csv_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if n.endswith('.csv')]
        if not members:
            raise FileNotFoundError(f"No CSV found in {zip_path}")
        name = members[0]
        with z.open(name) as f:
            raw = f.read()
            buf = io.BytesIO(raw)
            try:
                df = pd.read_csv(buf, header=None)
            except Exception:
                buf.seek(0)
                df = pd.read_csv(buf)
    if df.shape[1] >= 6:
        cols = ['id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch'][: df.shape[1]]
        df.columns = cols
    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    rename_map = {
        'isBuyerMaker': 'is_buyer_maker',
        'buyer_maker': 'is_buyer_maker',
        'bestMatch': 'is_best_match',
        'isBestMatch': 'is_best_match',
    }
    df = df.rename(columns=rename_map)
    required = ['time', 'price', 'qty', 'id', 'is_buyer_maker']
    if 'time' not in df.columns:
        for c in df.columns:
            if c.lower() in ('ts', 'timestamp'):
                df['time'] = df[c]
                break
    missing = [c for c in ['time', 'price', 'qty', 'id'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {zip_path}: {missing}")
    df['time'] = pd.to_numeric(df['time'], errors='coerce').astype('Int64').astype('int64')
    df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce').astype(float)
    df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64').astype('int64')
    if 'is_buyer_maker' in df.columns:
        if df['is_buyer_maker'].dtype != bool:
            df['is_buyer_maker'] = df['is_buyer_maker'].astype(str).str.lower().isin(['1', 'true', 't', 'yes'])
    else:
        df['is_buyer_maker'] = False
    df = df[required].sort_values('time')
    df.reset_index(drop=True, inplace=True)
    return df


# ------------------------- Processing (integrated 2h5) -------------------------

def _process_task(zip_file: str, from_date: Optional[str], dates: Optional[List[str]]):
    m = re.search(r"(\d{4}-\d{2})", os.path.basename(zip_file))
    if not m:
        return None
    date = m.group(1)
    if from_date and date < from_date:
        return None
    if dates and date not in dates:
        return None
    checksum_file = zip_file + '.CHECKSUM'
    print(f"\nProcessing {date}: {zip_file}")
    if not verify_zip_checksum(zip_file, checksum_file):
        raise ValueError(f"Checksum mismatch for {zip_file}. Please check the file integrity.")
    df = load_csv_from_zip(zip_file)
    trades = TradesData(
        df.time.values, df.price.values, df.qty.values, id=df.id.values,
        is_buyer_maker=df.is_buyer_maker.values,
        preprocess=True, name=os.path.basename(zip_file)
    )
    return trades, date


def _writer_thread(queue: multiprocessing.Queue, h5_path: str, counters: dict, lock: Lock):
    while True:
        item = queue.get()
        if item is None:
            break
        trades, date = item
        print(f"Writing {date} trades to {h5_path}...")
        trades.save_h5(h5_path, mode='a', overwrite_month=True)
        print(f"Finished writing {date} trades.")
        with lock:
            counters['consumed'] += 1


def process_all(root_dir: str, h5_path: str, from_date: Optional[str] = None, dates: Optional[List[str]] = None, workers: int = 4):
    zip_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.zip')])
    pbar = tqdm(total=len(zip_files), desc="Processing months")
    queue: multiprocessing.Queue = multiprocessing.Queue()
    counters = {'enqueued': 0, 'consumed': 0, 'errors': 0}
    lock = Lock()
    writer = Thread(target=_writer_thread, args=(queue, h5_path, counters, lock))
    writer.start()
    pool: Pool = Pool(workers)

    def on_done(output):
        if output:
            queue.put(output)
            with lock:
                counters['enqueued'] += 1
        pbar.update(1)
        with lock:
            backlog = counters['enqueued'] - counters['consumed']
            err = counters['errors']
        pbar.set_postfix(queue=backlog, errors=err)

    def on_error(err):
        print(f"Error processing file: {err}")
        with lock:
            counters['errors'] += 1
            backlog = counters['enqueued'] - counters['consumed']
            err_count = counters['errors']
        pbar.update(1)
        pbar.set_postfix(queue=backlog, errors=err_count)

    for zip_file in zip_files:
        pool.apply_async(_process_task, args=(zip_file, from_date, dates), callback=on_done, error_callback=on_error)
    pool.close()
    pool.join()

    pbar.close()
    queue.put(None)
    writer.join()


# ------------------------- Network & Orchestration -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def market_label(market: str) -> str:
    m = market.strip().lower()
    if m in ("spot",):
        return "SPOT"
    if m in ("um", "perpum", "perp_um", "usdm", "usdsm"):  # allow some aliases
        return "PERPum"
    if m in ("cm", "perpcm", "perp_cm", "coinm"):
        return "PERPcm"
    raise ValueError(f"Unknown market: {market}. Use spot, um, or cm.")


def market_path(market: str) -> str:
    m = market.strip().lower()
    if m == "spot":
        return "spot"
    if m in ("um", "perpum", "perp_um", "usdm", "usdsm"):
        return "um"
    if m in ("cm", "perpcm", "perp_cm", "coinm"):
        return "cm"
    raise ValueError(f"Unknown market: {market}.")


def build_urls(market: str, symbol: str, month: Month) -> Tuple[str, str]:
    mpath = market_path(market)
    if mpath == "spot":
        prefix = f"/data/spot/monthly/trades/{symbol}"
    else:
        prefix = f"/data/futures/{mpath}/monthly/trades/{symbol}"
    fname = f"{symbol}-trades-{month.year:04d}-{month.month:02d}.zip"
    url = BINANCE_BASE + f"{prefix}/{fname}"
    return url, url + ".CHECKSUM"


def download_file(url: str, dest_path: str, session: requests.Session, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    total=total if total > 0 else None, unit='B', unit_scale=True, desc=os.path.basename(dest_path), leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            if total > 0:
                                pbar.update(len(chunk))
            return True
        except Exception:
            if attempt == retries - 1:
                raise
    return False


def find_existing_h5(h5_dir: str, symbol: str, mlabel: str) -> List[str]:
    if not os.path.isdir(h5_dir):
        return []
    files: List[str] = []
    prefix = f"{symbol.upper()}_{mlabel}_"
    for name in os.listdir(h5_dir):
        if not name.lower().endswith('.h5'):
            continue
        if not name.upper().startswith(prefix.upper()):
            continue
        files.append(os.path.join(h5_dir, name))
    return sorted(files)


def parse_range_from_name(name: str) -> Tuple[str, str]:
    # SYMBOL_MARKET_YYMM-YYMM.h5 -> (YYMM, YYMM)
    m = re.search(r"_(\d{4})-(\d{4})\.h5$", name)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def orchestrate_symbol(symbol: str, market: str, start: Month, end: Month, workdir: str, workers: int, overwrite_klines: bool):
    mpath = market_path(market)
    raw_dir = os.path.join(workdir, 'raw', mpath, 'trades', symbol)
    h5_dir = os.path.join(workdir, 'h5')
    ensure_dir(raw_dir)
    ensure_dir(h5_dir)

    months = month_range(start, end)

    # Download all
    session = requests.Session()
    print(f"Downloading {symbol} [{market_label(market)}]: {months[0].ym()} -> {months[-1].ym()}")
    for mon in tqdm(months, desc=f"{symbol} monthly files"):
        url_zip, url_chk = build_urls(market, symbol, mon)
        file_zip = os.path.join(raw_dir, os.path.basename(url_zip))
        file_chk = file_zip + '.CHECKSUM'

        # Download ZIP if missing
        if not os.path.exists(file_zip):
            ok = download_file(url_zip, file_zip, session)
            if not ok:
                print(f"[warn] Not found {url_zip}; skipping month {mon.ym()}")
                continue
        # Download CHECKSUM if missing
        if not os.path.exists(file_chk):
            try:
                download_file(url_chk, file_chk, session)
            except Exception:
                pass
        # Verify checksum if possible
        if not verify_zip_checksum(file_zip, file_chk):
            print(f"[warn] Checksum mismatch: {file_zip}. Will re-download once.")
            try:
                if os.path.exists(file_zip):
                    os.remove(file_zip)
                download_file(url_zip, file_zip, session)
            except Exception:
                pass

    # Compute target h5 name with market label
    yymm_start = months[0].yymm()
    yymm_end = months[-1].yymm()
    mlabel = market_label(market)
    target_name = f"{symbol}_{mlabel}_{yymm_start}-{yymm_end}.h5"
    target_path = os.path.join(h5_dir, target_name)

    # Handle extension rename if an older file for same symbol+market exists
    existing = find_existing_h5(h5_dir, symbol, mlabel)
    for path in existing:
        a, b = parse_range_from_name(os.path.basename(path))
        if a == yymm_start and b and b < yymm_end:
            if path != target_path:
                print(f"Renaming {os.path.basename(path)} -> {target_name}")
                os.replace(path, target_path)
            break

    # Process raw zips into H5 for these months
    dates = [m.ym() for m in months]
    print(f"Processing {symbol} [{mlabel}] into {target_path}")
    process_all(raw_dir, target_path, from_date=None, dates=dates, workers=workers)

    # Add timebar klines
    print(f"Adding timebar klines for {symbol} [{mlabel}]...")
    tb = AddTimeBarH5(target_path)
    tb.process_all(overwrite=overwrite_klines)
    print(f"Finished {symbol}: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Binance trades -> H5 pipeline (Spot & Futures) with timebar klines")
    parser.add_argument('--market', required=True, help='Market: spot, um (USDⓈ-M), or cm (COIN-M). Also accepts SPOT, PERPum, PERPcm')
    parser.add_argument('--tickers', '-s', nargs='+', required=True, help='Symbols to download, e.g., BTCUSDT ETHUSDT')
    parser.add_argument('--start', required=True, help='Start date: YYYY or YYYY-MM')
    parser.add_argument('--end', default='now', help="End date: YYYY or YYYY-MM or 'now'")
    parser.add_argument('--workdir', required=True, help='Working directory for data (raw/ and h5/ will be created)')
    parser.add_argument('--workers', '-c', type=int, default=4, help='Workers for processing stage')
    parser.add_argument('--overwrite-klines', type=int, default=1, help='If 1, overwrite timebar klines during build')
    args = parser.parse_args()

    start_m = parse_ym(args.start)
    end_m = parse_ym(args.end)

    for sym in args.tickers:
        orchestrate_symbol(sym.upper(), args.market, start_m, end_m, args.workdir, args.workers, bool(args.overwrite_klines))


if __name__ == '__main__':
    main()
