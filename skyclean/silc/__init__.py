# skyclean/silc/__init__.py

__all__ = [
    "DownloadData",
    "FileTemplates",
    "HPTools",
    "MWTools",
    "SamplingConverters",
    "ProcessMaps",
    "Visualise",
    "SILCTools",
    "ProduceSILC",
    "utils",
]

# Map public names to (submodule, attribute)
_lazy = {
    "DownloadData": (".download", "DownloadData"),
    "FileTemplates": (".file_templates", "FileTemplates"),
    "HPTools": (".map_tools", "HPTools"),
    "MWTools": (".map_tools", "MWTools"),
    "SamplingConverters": (".map_tools", "SamplingConverters"),
    "ProcessMaps": (".map_processing", "ProcessMaps"),
    "Visualise": (".visualise", "Visualise"),
    "SILCTools": (".ilc", "SILCTools"),
    "ProduceSILC": (".ilc", "ProduceSILC"),
    "utils": (".utils", "utils"),  
}


def __getattr__(name):
    """Lazily import submodules and symbols on first access."""
    if name in _lazy:
        mod_path, attr_name = _lazy[name]
        module = __import__(__name__ + mod_path, fromlist=[attr_name])
        if name == "utils":
            # utils is a colection of functions, not a class within a module
            obj = module
        else:
            obj = getattr(module, attr_name)
        globals()[name] = obj  # Cache in module globals
        return obj
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Make dir() list all public symbols."""
    return sorted(list(globals().keys()) + __all__)
