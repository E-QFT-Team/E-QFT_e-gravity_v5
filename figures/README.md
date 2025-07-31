# Figures Directory

This directory will contain the output figures from running the physics tests.

To generate all figures, run:
```bash
python run_all_tests.py
```

Or generate individual figures:
- `python physics/generate_lr_plot.py` → lr_propagation_sigma4.png
- `python physics/lorentz_final.py` → lorentz_final.png
- `python physics/constraint_damping_simple.py` → constraint_evolution.png
- `python physics/waveform_validation.py` → waveform_comparison.png
- `python physics/matter_coupling_optimized.py` → klein_gordon_test.png, os_collapse_optimized.png
