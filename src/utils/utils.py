import pandas as pd
import numpy as np

def bool_to_int(x):
    """Convert boolean values to integers."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.astype(int)
    elif isinstance(x, bool):
        return int(x)
    elif isinstance(x, str) and x.lower() in ['true', 'yes', 'y', '1']:
        return 1
    elif isinstance(x, str) and x.lower() in ['false', 'no', 'n', '0']:
        return 0
    elif x is None:
        return 0
    else:
        return x