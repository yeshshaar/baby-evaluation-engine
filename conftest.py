import sys
import os

# Add project root to path so pytest can resolve `src.*` imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
