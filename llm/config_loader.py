import importlib.util

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file_path):
    """Loads configuration from a Python file into a Config object."""
    spec = importlib.util.spec_from_file_location("config_module", config_file_path)
    if spec is None:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return Config(config_module.__dict__)

_GLOBAL_CONFIG = None

def get_config():
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        raise RuntimeError("Configuration has not been loaded yet. Ensure load_config is called in main.py.")
    return _GLOBAL_CONFIG

def set_config(config_obj):
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config_obj