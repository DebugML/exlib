import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.append("../src")
import exlib
from exlib.datasets.cholec import get_cholec_scores
from exlib.datasets.chestx import get_chestx_scores
from exlib.datasets.mass_maps import get_mass_maps_scores
from exlib.datasets.supernova import get_supernova_scores
from exlib.datasets.multilingual_politeness import get_politeness_scores
from exlib.datasets.emotion import get_emotion_scores


all_settings_baselines = {
    'cholec': ['identity', 'random', 'patch', 'quickshift', 'watershed', 'sam', 'neural_quickshift', 'craft'],
    'chestx': ['identity', 'random', 'patch', 'quickshift', 'watershed', 'sam', 'neural_quickshift', 'craft'],
    'mass_maps': ['patch', 'quickshift', 'watershed', 'oracle', 'one'],
    'supernova': ['chunk 5', 'chunk 10', 'chunk 15'], # chunk size
    'multilingual_politeness': ['word', 'phrase', 'sentence'],
    'emotion': ['word', 'phrase', 'sentence']
}

all_settings_methods = {
    'cholec': get_cholec_scores,
    'chestx': get_chestx_scores,
    'mass_maps': get_mass_maps_scores,
    'supernova': get_supernova_scores,
    'multilingual_politeness': get_politeness_scores,
    'emotion': get_emotion_scores
}


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("setting")
    parser.add_argument("--results_dir", default=str(Path(Path(__file__).parent, "results")))
    args = parser.parse_args()
    print('SETTING:', args.setting)
    
    # Make output directory if does not exist
    output_dir = Path(args.results_dir, args.setting)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_filepath = str(Path(output_dir, f"all_baselines_scores.pt"))
    if os.path.isfile(scores_filepath):
        print(f'{scores_filepath} already exists')
    else:
        print('Getting scores...')
        # if len(sys.argv) > 1:
        baselines_list = all_settings_baselines[args.setting]
        get_scores = all_settings_methods[args.setting]

        all_baselines_scores = get_scores(baselines=baselines_list)
        # dic, where key is name of baseline (e.g. patch) and value is the baseline scores

        # print(all_baselines_scores)
        for name in all_baselines_scores:
            metric = torch.tensor(all_baselines_scores[name])
            mean_metric = metric.mean()
            print(f'BASELINE {name} mean score: {mean_metric}')
            
        torch.save(all_baselines_scores, scores_filepath)

        

