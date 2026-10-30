# System Architecture

## Data flow

```text
Solar panel + sensors
  voltage divider (A0), ACS712 (A1), LDR (A2), DHT22 (D2)
                 |
                 v
       firmware/solar_monitor.ino
       2-second serial CSV stream
                 |
                 v
       data_logger/logger.py
       parse -> validate -> accepted/rejected CSV
                 |
          +------+------+
          |             |
          v             v
analysis/analyze_     research/
performance.py        simulated anomaly benchmark
summary + plots       data prep, detectors, metrics
```

## Responsibilities and boundaries

- Firmware samples voltage, current, ambient temperature, humidity, and light.
- The Python logger timestamps receipt on the host, validates readings, and
  separates accepted rows from malformed or firmware-flagged rows.
- The analysis script calculates descriptive electrical summaries and plots.
- The research package compares range checks, a conditional residual detector,
  and Isolation Forest on a separately generated synthetic benchmark.
- The hardware team handled physical sensor setup and collection of real
  observations in the broader project. Raw/private hardware records are not
  stored in this public repository.

## Interpretation notes

- `timestamp_utc` is host receive time; `timestamp_ms` is Arduino uptime.
- `temp_c` is ambient DHT22 temperature, not panel temperature.
- `power_w` is calculated as voltage multiplied by current.
- An anomaly flag indicates a reading for review, not a fault diagnosis.
- Benchmark metrics describe injected synthetic patterns only. They do not
  establish real-world detector accuracy or predict equipment failure.
- Voltage-divider and ACS712 calibration constants are hardware-dependent
  placeholders and must be checked against calibrated instruments before use.
