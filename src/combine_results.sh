#!/bin/bash

# Define paths
RESULTS_DIR="../results"
COMBINED_RESULTS_FILE="$RESULTS_DIR/combined_results.csv"

# Combine all results into one CSV file
echo "Combining results..."

# Check if combined results file already exists and remove it
if [ -f "$COMBINED_RESULTS_FILE" ]; then
    rm "$COMBINED_RESULTS_FILE"
fi

# Get the header from the first file and write it to the combined file
first_file=$(ls $RESULTS_DIR/results-viking-*.csv | head -n 1)
if [ -f "$first_file" ]; then
    head -n 1 "$first_file" > "$COMBINED_RESULTS_FILE"
fi

# Loop through all results-viking-*.csv files and append to the combined file (skipping the header)
for result_file in $RESULTS_DIR/results-viking-*.csv; do
    echo "Adding $result_file to combined results..."
    tail -n +2 "$result_file" >> "$COMBINED_RESULTS_FILE"
done

echo "Results combined into $COMBINED_RESULTS_FILE"
