# Try to import bindings
try:
    from . import avatars_bindings
except:
    import warnings
    warnings.warn("Failed to import avatars_bindings module", ImportWarning)
    mesh = None