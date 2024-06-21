import numpy as np
import pandas as pd
import os
from huggingface_hub import hf_hub_download

def load_lexica(language):
    REPO_ID = "BrachioLab/multilingual_politeness_helper"
    FILENAME = "{}_politelex.csv".format(language)
    return pd.read_csv(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset"))
