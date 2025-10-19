# tests/__init__.py

__all__ = [
    "Tests",
]

# Map public names to (module, attribute)
_lazy = {
    "Tests": (".test_skyclean", "Tests"),
}

def __getattr__(name):
    """Lazily import submodules and symbols on first access."""
    if name in _lazy:
        mod_path, attr_name = _lazy[name]
        module = __import__(__name__ + mod_path, fromlist=[attr_name])
        obj = getattr(module, attr_name)
        globals()[name] = obj  # Cache for next time
        return obj
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Ensure dir() shows all public names."""
    return sorted(list(globals().keys()) + __all__)
