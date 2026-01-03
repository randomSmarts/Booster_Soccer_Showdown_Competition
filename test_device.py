import torch
import sys
import os

# Add training_scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_scripts'))

try:
    from sac import SAC
    print("Successfully imported SAC from training_scripts/sac.py")
except ImportError as e:
    print(f"Failed to import SAC: {e}")
    sys.exit(1)

# Mock env properties
class MockActionSpace:
    shape = (3,)

def test_device_selection():
    print(f"Torch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Test auto-detection
    agent = SAC(n_features=10, action_space=MockActionSpace())
    print(f"Auto-detected device: {agent.device}")

    # Test explicit MPS
    if torch.backends.mps.is_available():
        agent_mps = SAC(n_features=10, action_space=MockActionSpace(), device="mps")
        print(f"Explicit MPS device: {agent_mps.device}")
        assert agent_mps.device.type == "mps"

if __name__ == "__main__":
    test_device_selection()

