# Force MPL to use non-gui backends for testing.
import matplotlib

try:
    pass
except ImportError:
    pass
else:
    matplotlib.use('Agg')
