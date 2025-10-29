import gnx.core

gnx.core.enable_jax_backend()

print("Backends", gnx.core.backends())
print("Devices", gnx.core.devices())
