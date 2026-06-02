# Task 8: Dynamical Decoupling

This task rebuilds the single-qubit dynamical decoupling examples on top of the
current MuSIQ resource schema.

Included sequences:

- `fid`: free induction decay baseline
- `hahn`: single refocusing pulse
- `cpmg`: Carr-Purcell-Meiboom-Gill
- `xy4`: XY-4 decoupling
- `xy8`: XY-8 decoupling
- `udd`: Uhrig dynamical decoupling

Notes:

- Circuit files use the canonical OpenQASM 3 resource schema.
- Canonical single-qubit gate names (`x`, `y`) are used so the bundle matches
  the naming style of the other MuSIQ examples.
- The DD sequences are wrapped by `sx ... sx` so the comparison is sensitive to
  phase noise in the equatorial plane instead of only tracking basis
  population from `|0>`.
- Unequal idle segments are preserved via custom pulse recipes such as
  `wait_2500` and `wait_6180`.
- The OU source in `device.yaml` is tuned as a slow frequency-drift channel,
  with correlation time much longer than the interpulse spacing, so Hahn /
  CPMG / XY sequences can visibly refocus it.
- `calibrate.py` fits `x` and `sx` separately in the three-level transmon model
  and writes the recommended amplitudes to `calibration_results.json`.
