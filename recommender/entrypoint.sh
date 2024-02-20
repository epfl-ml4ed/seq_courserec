#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode.
set -euo pipefail
# ... Run whatever commands ...

# Temporarily disable strict mode and activate conda:
set +euo pipefail
conda activate pgpr

# Re-enable strict mode:
set -euo pipefail

# exec the final command:
#python preprocess.py $CONFIG
#python train_transe_model.py $CONFIG
python train_agent.py $CONFIG
python test_agent.py $CONFIG