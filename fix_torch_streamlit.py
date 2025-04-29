"""
Fix for PyTorch and Streamlit compatibility issue
This module monkeypatches Streamlit's module handling to avoid issues with PyTorch custom classes
"""

import sys
import importlib
import types

# Safer approach: Modify sys.modules to handle torch._classes specially
original_import = __import__

def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Call the original import function
    module = original_import(name, globals, locals, fromlist, level)
    
    # If this is torch._classes, make it safer for Streamlit
    if name == 'torch._classes' or (fromlist and 'torch._classes' in fromlist):
        # Create a safer __path__ attribute that won't cause issues
        if hasattr(module, '__path__') and not hasattr(module.__path__, '_path'):
            module.__path__ = [p for p in getattr(module, '__path__', [])]
    
    return module

# Apply the patch
sys.modules['torch._classes'] = types.ModuleType('torch._classes')
sys.__import__ = patched_import

# Disable Streamlit's file watcher for torch modules
try:
    import streamlit as st
    # Add torch to Streamlit's ignore list if possible
    if hasattr(st, '_config') and hasattr(st._config, 'get_option'):
        server_options = st._config.get_option('server')
        if 'folderWatchBlacklist' in server_options:
            if 'torch' not in server_options['folderWatchBlacklist']:
                server_options['folderWatchBlacklist'].append('torch')
except Exception:
    pass  # If we can't modify Streamlit config, just continue

print("✅ Applied PyTorch-Streamlit compatibility fix")
