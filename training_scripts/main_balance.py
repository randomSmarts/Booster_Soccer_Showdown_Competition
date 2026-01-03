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
    train_balance()
    print("Balance training finished.")