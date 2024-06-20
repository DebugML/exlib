import numpy as np
import pandas as pd
import os

# from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

def load_lexica(language):
    parent_dir = Path(__file__).parent

    REPO_ID = "BrachioLab/multilingual_politeness_helper"
    FILENAME = "{}_politelex.csv".format(language)
    return pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset"))
    
    # return pd.read_csv(parent_dir / "politeness_lexica/{}_politelex.csv".format(language))
