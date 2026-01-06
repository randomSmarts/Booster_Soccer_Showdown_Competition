# get_spawn.py
import gymnasium as gym
import numpy as np
import inspect
import sai_mujoco.utils.v0.binding_utils as bu

# Makes sure it doesn't crash if there is a name typo
def make_patched_method(original_method):
    def patched(self, name):
        try: return original_method(self, name)
        except ValueError:
            try: return original_method(self, f"/{name}")
            except ValueError: raise
    return patched

# Finds the class responsible for naming things
TargetClass = None
for name, obj in inspect.getmembers(bu):
    if inspect.isclass(obj) and hasattr(obj, 'body_name2id'):
        TargetClass = obj
        break
if TargetClass: #applies the patch method to all of the naming stuff
    for m in ['body_name2id', 'geom_name2id', 'joint_name2id', 'site_name2id', 'sensor_name2id', 'actuator_name2id']:
        if hasattr(TargetClass, m): setattr(TargetClass, m, make_patched_method(getattr(TargetClass, m)))

# Setups the specified environment with everything at default values
ENV_ID = "LowerT1GoaliePenaltyKick-v0"
print(f"Resetting {ENV_ID}...")

env = gym.make(ENV_ID)
env.reset()

# Unwrap to get raw data
base = env.unwrapped #gets the raw MuJoCo Engine
if hasattr(base, "model"): model, data = base.model, base.data #model is the rules of the world and data is the current state
elif hasattr(base, "sim"): model, data = base.sim.model, base.sim.data
else: raise ValueError("Could not access MuJoCo")

# Get raw pointers if needed (SAI wrappers)
if hasattr(data, "ptr"): data = data.ptr
elif hasattr(data, "_data"): data = data._data

# The robot starts at index 9. Everything before that is World/Ball.
print("\n--- COPY THESE NUMBERS ---")
ball_spawn = data.qpos[:9].copy() #data.qpos represents the starting position of all items, it then grabs the first 9 numbers
print(f"STATIC_SPAWN = {list(ball_spawn)}")
print("--------------------------")