#!/bin/bash

# OpenFOAM Channel Flow Simulation Script
# This script runs the complete OpenFOAM simulation for k-epsilon channel flow

set -e  # Exit on any error

# Check if OpenFOAM is sourced
if [ -z "$WM_PROJECT" ]; then
    echo "Error: OpenFOAM environment not found."
    echo "Please source OpenFOAM before running this script:"
    echo "source /opt/openfoam9/etc/bashrc"
    exit 1
fi

echo "=========================================="
echo "OpenFOAM Channel Flow k-epsilon Simulation"
echo "=========================================="
echo "OpenFOAM version: $WM_PROJECT_VERSION"
echo "Case directory: $(pwd)"
echo "Start time: $(date)"
echo "=========================================="

# Clean previous results
echo "Cleaning previous results..."
./Allclean 2>/dev/null || {
    echo "Allclean script not found, manually cleaning..."
    rm -rf [0-9]* processor* postProcessing constant/polyMesh log.* *.log
}

# Generate mesh
echo "Generating mesh with blockMesh..."
blockMesh > log.blockMesh 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Mesh generation completed successfully"
else
    echo "✗ Mesh generation failed. Check log.blockMesh"
    exit 1
fi

# Check mesh quality
echo "Checking mesh quality..."
checkMesh > log.checkMesh 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Mesh quality check passed"
else
    echo "⚠ Warning: Mesh quality issues detected. Check log.checkMesh"
fi

# Run the simulation
echo "Running simpleFoam solver..."
echo "This may take several minutes..."

# Run with time monitoring
start_time=$(date +%s)
simpleFoam > log.simpleFoam 2>&1 &
solver_pid=$!

# Monitor progress
while kill -0 $solver_pid 2>/dev/null; do
    if [ -f log.simpleFoam ]; then
        last_time=$(tail -20 log.simpleFoam | grep "^Time = " | tail -1 | awk '{print $3}')
        if [ ! -z "$last_time" ]; then
            echo -ne "\rSolver running... Current time: $last_time"
        fi
    fi
    sleep 5
done

wait $solver_pid
solver_exit_code=$?
echo ""  # New line after progress monitoring

end_time=$(date +%s)
runtime=$((end_time - start_time))

if [ $solver_exit_code -eq 0 ]; then
    echo "✓ simpleFoam completed successfully in ${runtime} seconds"
else
    echo "✗ simpleFoam failed. Check log.simpleFoam"
    exit 1
fi

# Post-processing
echo "Running post-processing..."

# Check if functions are executed (sampling, etc.)
if grep -q "samplingSets" system/controlDict; then
    echo "✓ Sampling data collected during simulation"
fi

# Generate additional post-processing if needed
echo "Generating additional output data..."

# Extract data at final time step for PINN comparison
final_time=$(ls -1 [0-9]* | sort -n | tail -1)
echo "Final time step: $final_time"

# Convert OpenFOAM data to CSV format for PINN comparison
if [ -d "postProcessing/samplingSets/$final_time" ]; then
    echo "Converting sampling data to CSV format..."
    
    # Find the sampling file
    sample_file=$(find postProcessing/samplingSets/$final_time -name "*.csv" | head -1)
    if [ ! -z "$sample_file" ]; then
        # Copy to a standardized location
        cp "$sample_file" "../../data/openfoam_reference_data.csv"
        echo "✓ Reference data saved to ../../data/openfoam_reference_data.csv"
    fi
fi

# Copy the main output data if it exists
if [ -f "output_data.csv" ]; then
    cp "output_data.csv" "../../data/openfoam_output_data.csv"
    echo "✓ Output data copied to ../../data/"
fi

# Summary
echo "=========================================="
echo "Simulation Summary:"
echo "- Runtime: ${runtime} seconds"
echo "- Final time: $final_time"
echo "- Mesh cells: $(grep -E "cells:" log.checkMesh | tail -1 | awk '{print $2}' || echo 'N/A')"
echo "- Convergence: $(tail -10 log.simpleFoam | grep -E "Final residual|converged" | wc -l) equations converged"
echo "=========================================="

# Check for convergence
if grep -q "SIMPLE solution converged" log.simpleFoam; then
    echo "✓ Solution converged successfully"
elif grep -q "End" log.simpleFoam; then
    echo "⚠ Solution completed but may not be fully converged"
    echo "  Check residuals in log.simpleFoam"
else
    echo "✗ Solution may not have completed properly"
    exit 1
fi

echo "Simulation completed at: $(date)"
echo "Results are available in time directories: $(ls -1 [0-9]* | tr '\n' ' ')"

# Cleanup temporary files
echo "Cleaning up temporary files..."
rm -f dynamicCode/platforms/*/lib/lib* 2>/dev/null || true

echo "=========================================="
echo "OpenFOAM simulation completed successfully!"
echo "Use 'paraview' to visualize results"
echo "PINN comparison data saved to ../../data/"
echo "=========================================="
