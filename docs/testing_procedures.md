# Testing and Calibration Procedures

## Software tests

From the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover tests
# or
pytest tests/ -v
```

The logger tests exercise parsing, range validation, and accepted/rejected CSV
logging without requiring a connected board. Research tests cover data quality,
chronological splitting, causal feature construction, synthetic fault labels,
detector calibration, and evaluation metrics.

## Bench calibration before connecting a panel

1. Confirm the board, sensor modules, wiring, and panel ratings.
2. Compare the voltage-divider measurement with a suitable multimeter and
   calculate `VOLTAGE_DIVIDER_RATIO`.
3. Calibrate the ACS712 zero-current offset and sensitivity for the exact
   module variant; do not assume every board uses the example 5 A sensitivity.
4. Compare DHT22 readings with a reference thermometer/hygrometer.
5. Check the LDR response at several light levels; it reports relative
   percentage, not calibrated irradiance in W/m².
6. Verify the documented eight-field serial output at 9600 baud.
7. Confirm valid rows enter the accepted CSV and malformed, sensor-fault, and
   out-of-range rows enter the rejects CSV with a reason.

## Field-data handling

- Record hardware configuration and calibration values for each session.
- Keep raw/private hardware records outside this public repository unless
  sharing is explicitly approved.
- Review timestamps, missing samples, duplicates, sentinels, status values,
  and power consistency before analysis.
- Do not label a reading a confirmed fault without independent inspection or
  maintenance evidence.
- The committed ML benchmark is synthetic; do not use its scores as a field
  acceptance criterion.
