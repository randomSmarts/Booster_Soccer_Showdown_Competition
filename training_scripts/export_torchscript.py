import torch
import os
from sac import SAC
from main import ENV_IDS, Preprocessor, MultiTaskEnv

def export():
    env = MultiTaskEnv(ENV_IDS)
    preprocessor = Preprocessor()

    obs, info = env.reset()
    n_features = preprocessor.modify_state(obs, info).shape[1]

    agent = SAC(
        n_features=n_features,
        action_space=env.action_space,
        log_std_min=-4.0,
        log_std_max=-0.5,
        alpha=0.03,
        alpha_decay=0.9999,
        alpha_min=0.01,
        device="cpu",
    )

    ckpt = "balance_models/sac_balance_best.pth"
    agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.eval()

    scripted = torch.jit.script(agent)
    scripted.save("balance_models/sac_balance_best.pt")

    print("saved TorchScript model")

if __name__ == "__main__":
    export()