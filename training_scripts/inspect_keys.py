# Usage: python inspect_keys.py demonstrations/goal_kick.npz
# Understanding file structure of goal_kick.npz=

import numpy as np
import sys

filename = sys.argv[1]
data = np.load(filename)

print(f"--- Keys in {filename} ---")
print(list(data.keys()))

# Optional: Print shape of likely candidates to find the actions
for key in data.keys():
    obj = data[key]
    if hasattr(obj, 'shape'):
        print(f"{key}: {obj.shape}")

# this file is used to determine STATE_DIM and ACTION_DIM since you load in a specific demo file and then extract the labels of the data stored inside