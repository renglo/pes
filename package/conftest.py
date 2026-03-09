import sys
import os

# Add noma extension package to sys.path so tests can import noma.handlers.*
_noma_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'noma', 'package'))
if os.path.isdir(_noma_pkg) and _noma_pkg not in sys.path:
    sys.path.insert(0, _noma_pkg)
