"""
Root conftest.py
Adds the project root to sys.path so that top-level packages
(db, gateway, config, etc.) are importable when running pytest
from the project root.
"""
import sys
import os

# Insert the project root (the directory this file lives in) at the
# front of sys.path so 'db', 'gateway', 'config', etc. are importable.
sys.path.insert(0, os.path.dirname(__file__))