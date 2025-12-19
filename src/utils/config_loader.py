"""
Config loader utility for loading configuration files
"""
import json
import os
from pathlib import Path


def load_config(config_name):
    """
    Load configuration from config directory
    
    Args:
        config_name: Name of config file (with or without .json extension)
        
    Returns:
        dict: Configuration dictionary
    """
    if not config_name.endswith('.json'):
        config_name = f"{config_name}.json"
    
    config_path = Path(__file__).parent.parent / 'config' / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def get_path(path_key, subkey=None):
    """
    Get path from paths.json config
    
    Args:
        path_key: Main key in paths.json (e.g., 'data', 'embeddings', 'output')
        subkey: Optional subkey for nested paths
        
    Returns:
        str: Path string
    """
    paths_config = load_config('paths.json')
    
    if subkey:
        return paths_config[path_key][subkey]
    else:
        return paths_config[path_key]


def resolve_path(path_str):
    """
    Resolve relative path to absolute path from project root
    
    Args:
        path_str: Relative path string
        
    Returns:
        Path: Absolute Path object
    """
    project_root = Path(__file__).parent.parent
    return project_root / path_str


def ensure_dir(path):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path (string or Path object)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
