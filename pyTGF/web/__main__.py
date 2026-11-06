"""Allow `python -m pyTGF.web` to launch the web server."""
from .server import main
import sys
sys.exit(main())
