import sys
import os

# This is the balance and recovery training to just get the robot to balance 

# Add root directory to path to find sai_patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add the current directory to path so imports work when running from inside training_scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sai_patch #required for the naming fixes
    print("[main_balance] sai_patch imported.")
except ImportError:
    print("[main_balance] Warning: sai_patch not found.")

from sai_rl import SAIClient

from train_balance import train_balance

if __name__ == "__main__":
    # Optional: initialize SAI (not required for local training loop)
    try:   
        _sai = SAIClient(
            comp_id="lower-t1-penalty-kick-goalie",
            api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj",
        )
    except Exception:
        print("SAIClient init failed (ok for local training).")

    print("Starting Stage 1.5 Training: Balance Recovery (SAC)")
    train_balance() #calls the training function
    print("Balance training finished.")

# Episodes: How many times the robot has died and restarted
# Avg(20): Represents the average reward across the past 20 episodes (it going up is better)
# BestEp: Max reward out of all past episodes
# EpSteps: How many frames robot stayed alive in current try
# TotalSteps: Total amount of experience the robot has collected across all tries