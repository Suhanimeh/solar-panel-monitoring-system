"""Host-side serial parser and CSV logger."""
from __future__ import annotations
import argparse, csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

VALID_RANGES = {
    "voltage_v": (0.0, 24.0), "current_a": (-1.0, 10.0),
    "temp_c": (-20.0, 60.0), "humidity_pct": (0.0, 100.0),
    "light_pct": (0.0, 100.0),
}
CSV_FIELDS = ["timestamp_utc", "timestamp_ms", "voltage_v", "current_a",
              "power_w", "temp_c", "humidity_pct", "light_pct", "status"]

class ValidationError(ValueError):
    pass

@dataclass
class Reading:
    timestamp_utc: str
    timestamp_ms: int
    voltage_v: float
    current_a: float
    power_w: float
    temp_c: float
    humidity_pct: float
    light_pct: float
    status: str

def parse_line(line: str) -> Reading:
    fields = line.strip().split(",")
    if len(fields) != 8:
        raise ValidationError(f"expected 8 fields, received {len(fields)}")
    try:
        timestamp_ms = int(fields[0])
        values = [float(v) for v in fields[1:7]]
    except ValueError as exc:
        raise ValidationError("non-numeric field in serial row") from exc
    status = fields[7].strip()
    if status not in {"OK", "SENSOR_FAULT", "OUT_OF_RANGE"}:
        raise ValidationError(f"unknown firmware status: {status or '<empty>'}")
    reading = Reading(datetime.now(timezone.utc).isoformat(), timestamp_ms, *values, status)
    if status == "OK":
        validate_ranges(reading)
    return reading

def validate_ranges(reading: Reading) -> None:
    if reading.status != "OK":
        return
    for field, (lo, hi) in VALID_RANGES.items():
        value = getattr(reading, field)
        if not lo <= value <= hi:
            raise ValidationError(f"{field}={value} outside valid range [{lo}, {hi}]")

class DataLogger:
    def __init__(self, outdir):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.good_path = self.outdir / f"solar_readings_{stamp}.csv"
        self.reject_path = self.outdir / f"solar_rejects_{stamp}.csv"
        self.accepted = self.rejected = 0
        with self.good_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        with self.reject_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_utc", "raw_line", "reason"])

    def log_good(self, reading):
        validate_ranges(reading)
        with self.good_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(asdict(reading))
        self.accepted += 1

    def log_reject(self, raw_line, reason):
        with self.reject_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), raw_line, reason])
        self.rejected += 1

    def summary(self):
        total = self.accepted + self.rejected
        return f"{self.accepted} accepted, {self.rejected} rejected, {total} rows"

def main():
    parser = argparse.ArgumentParser(description="Read solar-monitor serial CSV data")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--outdir", type=Path, default=Path("../data"))
    args = parser.parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial with: pip install pyserial") from exc
    logger = DataLogger(args.outdir)
    print(f"Accepted rows: {logger.good_path}; rejected rows: {logger.reject_path}")
    try:
        with serial.Serial(args.port, args.baud, timeout=2) as connection:
            while True:
                raw = connection.readline().decode("utf-8", errors="replace").strip()
                if not raw or raw.startswith("timestamp_ms"):
                    continue
                try:
                    reading = parse_line(raw)
                    if reading.status == "OK":
                        logger.log_good(reading)
                    else:
                        logger.log_reject(raw, f"firmware status: {reading.status}")
                except ValidationError as exc:
                    logger.log_reject(raw, str(exc))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(logger.summary())

if __name__ == "__main__":
    main()
