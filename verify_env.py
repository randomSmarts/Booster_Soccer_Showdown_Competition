import sys

def check_import(module_name):
    try:
        module = __import__(module_name)
        print(f"✅ {module_name:<15} : {getattr(module, '__version__', 'installed')}")
        return module
    except ImportError as e:
        print(f"❌ {module_name:<15} : Failed ({e})")
        return None

print(f"Python executable: {sys.executable}")
print("-" * 40)

# Core ML libs
torch = check_import('torch')
if torch:
    print(f"   -> CUDA available: {torch.cuda.is_available()}")

jax = check_import('jax')
if jax:
    try:
        print(f"   -> JAX devices: {jax.devices()}")
    except Exception as e:
        print(f"   -> JAX device check failed: {e}")

# Simulation & SAI libs
mujoco = check_import('mujoco')
sai_mujoco = check_import('sai_mujoco')
sai_rl = check_import('sai_rl')

print("-" * 40)
print("Environment verification complete.")

