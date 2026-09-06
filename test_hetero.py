import gymnasium as gym
import mani_skill.envs
import numpy as np

env = gym.make("OpenCabinetDrawer-v1", num_envs=4, robot_uids="fetch")
print(f"Num envs: {env.num_envs}")
print(f"Reconfiguration freq: {env.reconfiguration_freq}")

# Check if model_ids are different
# We might need to access internal state as it's not easily visible from obs
# But we can check env.unwrapped._cabinets or similar if we can access it.

# Accessing private attribute for debugging
try:
    envs = env.unwrapped
    # envs is the OpenCabinetDrawerEnv instance
    
    # Check what model_ids were used.
    # We can't see the local variable `model_ids` from _load_cabinets, 
    # but we can check the names of the cabinets loaded.
    
    cabinets = envs._cabinets
    print(f"Loaded {len(cabinets)} cabinets")
    names = [c.name for c in cabinets]
    print(f"Cabinet names: {names}")
    
    unique_names = set([n.split('-')[0] for n in names]) # name is model_id-i
    print(f"Unique model IDs: {unique_names}")
    
except Exception as e:
    print(f"Error inspecting env: {e}")

env.close()
