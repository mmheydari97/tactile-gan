#!/bin/bash

# Find all scripts in the current directory
for script in *.sh; do

  # Check if the script is not the current script
  if [ "$script" != "run_exp.sh" ]; then

    # Run the script with a 5 minute gap
    echo "Running script: $script"
    sbatch "$script"
    sleep 10s
  fi

done
