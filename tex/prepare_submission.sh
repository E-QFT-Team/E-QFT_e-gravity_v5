#!/bin/bash
# Prepare submission package for journal

# Create submission directory
mkdir -p submission/figures

# Copy LaTeX file
cp full_gr.tex submission/

# Copy all figures to figures subdirectory
cp ../physics/lr_propagation_sigma4.png submission/figures/
cp ../physics/lorentz_final.png submission/figures/
cp ../physics/constraint_evolution.png submission/figures/
cp ../physics/waveform_comparison.png submission/figures/
cp ../physics/klein_gordon_test.png submission/figures/
cp ../physics/os_collapse_optimized.png submission/figures/
cp ../physics/lr_continuum_limit.png submission/figures/

# Create submission archive
cd submission
zip -r ../eqft_full_gr_submission.zip *
cd ..

echo "Submission package created: eqft_full_gr_submission.zip"
echo "Contents:"
echo "  - full_gr.tex"
echo "  - figures/ (all PNG files)"