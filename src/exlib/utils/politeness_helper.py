import numpy as np
import pandas as pd
import os

from pathlib import Path

def load_lexica(language):
    parent_dir = Path(__file__).parent
    return pd.read_csv(parent_dir / "politeness_lexica/{}_politelex.csv".format(language))
