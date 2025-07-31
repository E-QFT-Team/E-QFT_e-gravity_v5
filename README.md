# E-QFT v5.0: Full General Relativity from Emergent Quantum Field Theory

[![Version](https://img.shields.io/badge/version-5.0-blue.svg)](https://github.com/eqft-collaboration/egravity-v5)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16642478-blue.svg)](https://doi.org/10.5281/zenodo.16642478)

## Overview

This repository contains the complete implementation of E-QFT (Emergent Quantum Field Theory) v5.0, which demonstrates the emergence of full General Relativity from quantum projector fields. Starting from quantum information-theoretic principles, we recover:

- **Causal structure**: Light cones with Lieb-Robinson velocity v_LR → c
- **Lorentz invariance**: Systematically vanishing violations α_n ∝ a^(n-2)
- **Einstein equations**: Via stable BSSN evolution
- **Gravitational waves**: Matching numerical relativity to < 10^-14
- **Matter coupling**: Klein-Gordon fields and Oppenheimer-Snyder collapse

## Key Results

| Test | Result | Publication Figure |
|------|--------|-------------------|
| Lieb-Robinson velocity | v_LR = (0.96 ± 0.01)c | Fig. 1 |
| Lorentz violation | α₄ = (σa)²/12 → 0 | Fig. 2 |
| BSSN constraints | \|\|H\|\|₂ < 4×10⁻⁴ | Fig. 3 |
| Waveform mismatch | < 10⁻¹⁴ | Fig. 4 |
| OS collapse error | 9.9% → 0% as σ/a → ∞ | Fig. 6 |

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4
- Numba >= 0.54
- h5py >= 3.0

### Setup

```bash
git clone https://github.com/eqft-collaboration/egravity-v5.git
cd egravity-v5
pip install -r requirements.txt
```

## Quick Start

### Run All Tests

```bash
python run_all_tests.py
```

This will execute all priority tests and generate:
- `test_summary.png` - Visual test results
- `test_results.json` - Detailed test data
- `PUBLICATION_CHECKLIST.txt` - Publication readiness

### Individual Priority Tests

```python
# Priority 1: Lieb-Robinson velocity
from physics.lr_measurement_final import measure_lieb_robinson_velocity
v_LR, v_LR_err = measure_lieb_robinson_velocity(L=32, sigma=4.0)

# Priority 2: Lorentz invariance
from physics.lorentz_final import test_lorentz_invariance
results = test_lorentz_invariance()

# Priority 3: BSSN constraints
from physics.constraint_damping_simple import main
main()  # Runs full constraint analysis

# Priority 4: Waveform validation
from physics.waveform_validation import validate_waveforms
mismatch = validate_waveforms()

# Priority 5: Matter coupling
from physics.matter_coupling_optimized import test_klein_gordon
kg_passed = test_klein_gordon()
```

## Repository Structure

```
V5.0/
├── core/                    # Core 4D implementation
│   ├── projectors_4d.py    # 4D covariant projectors
│   ├── symplectic_4d.py    # Symplectic evolution
│   └── adm_evolution.py    # ADM formalism
├── geometry/               # Geometric structures
│   ├── dec_4d.py          # Discrete exterior calculus
│   └── bssn.py            # BSSN evolution
├── physics/               # Physics tests & validation
│   ├── lr_measurement_final.py      # Priority 1
│   ├── lorentz_final.py            # Priority 2
│   ├── constraint_damping_simple.py # Priority 3
│   ├── waveform_validation.py      # Priority 4
│   └── matter_coupling_optimized.py # Priority 5
├── tex/                   # LaTeX publication
│   └── full_gr.tex       # Main manuscript
└── run_all_tests.py      # Master test suite
```

## Reproducing Publication Results

### Figure 1: Lieb-Robinson Light Cone
```bash
python physics/generate_lr_plot.py
# Output: lr_propagation_sigma4.png
```

### Figure 2: Lorentz Invariance
```bash
python physics/lorentz_final.py
# Output: lorentz_final.png
```

### Figure 3: BSSN Constraints
```bash
python physics/constraint_damping_simple.py
# Output: constraint_evolution.png
```

### Figure 4: Waveform Comparison
```bash
python physics/waveform_validation.py
# Output: waveform_comparison.png
```

### Figure 5: Klein-Gordon Field
```bash
python physics/matter_coupling_optimized.py test_klein_gordon
# Output: klein_gordon_test.png
```

### Figure 6: Oppenheimer-Snyder Collapse
```bash
python physics/matter_coupling_optimized.py
# Output: os_collapse_optimized.png
```

## Theoretical Background

E-QFT posits that spacetime emerges from quantum projector overlaps:

```
g_μν(x) = Σ_αβ c_α c*_β ⟨P_α|ĝ_μν|P_β⟩ exp(-|x-x_α|²/4σ²) exp(-|x-x_β|²/4σ²)
```

where:
- `|P_α⟩` are 4D covariant projector states
- `σ` is the projector width
- `c_α` are quantum amplitudes

The continuum limit (a → 0, σ/a fixed) recovers exact GR.

## Performance Notes

- Default lattice size: L = 32³
- Typical runtime: ~10 min for full test suite
- Memory usage: ~2 GB for L = 32³
- GPU acceleration: Planned for v6.0

## Citation

If you use this code, please cite:

```bibtex
@article{EQFT2025,
  title={General Relativity as an Emergent Limit of Quantum Projector Fields},
  author={Barreiro, Lionel and others},
  collaboration={E-QFT},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

## License

This project is licensed under GPL-3.0. See [LICENSE](LICENSE) for details.

## Contact

- **Corresponding Author**: Lionel Barreiro (lionel.barreiro@eqft-institute.org)
- **GitHub Issues**: https://github.com/E-QFT-Team/e-gravity-v5/issues

## Acknowledgments

We thank the E-QFT collaboration for valuable discussions. Computations were performed using the E-QFT v5.0 framework. This work was supported by [funding sources].

---

**Note**: This is research code accompanying a scientific publication. While we strive for correctness, use at your own risk. Please report any issues.
