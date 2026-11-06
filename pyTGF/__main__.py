"""Allow `python -m pyTGF` to invoke main.py"""
from .main import main
import sys
sys.exit(main())
