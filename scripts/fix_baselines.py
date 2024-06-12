import argparse
import os.path
import torch
import sys

# sys.path.append("../src/exlib/features/fix")
sys.path.append("../src")
from exlib.features.vision.cholec import get_cholec_scores
from exlib.features.vision.chestx import get_chestx_scores
sys.path.append(".")
from fix_massmaps import get_massmaps_scores
from exlib.features.time_series.supernova import get_supernova_scores
from exlib.features.text.politeness import get_politeness_scores
# from exlib.features.text.emotion import get_emotion_scores


all_settings_baselines = {
    'cholec': ['patch', 'quickshift', 'watershed'],
    'chestx': ['patch', 'quickshift', 'watershed'],
    'massmaps': ['patch', 'quickshift', 'watershed'],
    'supernova': ['chunk'],
    'politeness': ['word', 'phrase', 'sentence'],
    # 'emotion': ['word', 'phrase', 'sentence']
}

all_settings_methods = {
    'cholec': get_cholec_scores,
    'chestx': get_chestx_scores,
    'massmaps': get_massmaps_scores,
    'supernova': get_supernova_scores,
    'politeness': get_politeness_scores,
    # 'emotion': get_emotion_scores
}


if __name__ == "__main__": 
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("setting")
    # if len(sys.argv) > 1:
    #     parser.add_argument("baselines_list")
    args = parser.parse_args()

    print('SETTING:', args.setting)
    scores_filepath = f'results/{args.setting}/all_fix_baselines_scores' # need to have these results folders set up
    if os.path.isfile(scores_filepath):
        print(f'{scores_filepath} already exists')
    else:
        print('Getting scores...')
        # if len(sys.argv) > 1:
        baselines_list = all_settings_baselines[args.setting]
        get_scores = all_settings_methods[args.setting]
    
        all_baselines_scores = get_scores(baselines_list)
        # dic, where name (patch) of BL and value is the patch BL number
        
        print(all_baselines_scores)
        torch.save(all_baselines_scores, scores_filepath)
    