#!/usr/bin/env python3
"""
Quick start example for E-QFT v5.0

This demonstrates the key functionality of the package.
"""

import sys
sys.path.append('..')

# Example 1: Measure Lieb-Robinson velocity
print("Example 1: Measuring Lieb-Robinson velocity...")
from physics.lr_measurement_final import measure_lieb_robinson_velocity

v_LR, v_LR_err = measure_lieb_robinson_velocity(
    L=16,  # Smaller lattice for quick demo
    sigma=4.0,
    T_max=10.0,
    dt=0.1
)
print(f"Lieb-Robinson velocity: v_LR = ({v_LR:.3f} ± {v_LR_err:.3f})c")

# Example 2: Test Klein-Gordon field
print("\nExample 2: Testing Klein-Gordon field...")
from physics.matter_coupling_optimized import test_klein_gordon

result = test_klein_gordon()
print(f"Klein-Gordon test: {'PASSED' if result else 'FAILED'}")

print("\nFor full test suite, run: python run_all_tests.py")
