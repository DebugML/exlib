import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .emotion_helper import project_points_onto_axes, line_intersection, load_emotions
from .politeness_helper import load_lexica

def hatch_dim_and_outline(image, mask, c=0.6): 
    black = np.zeros(image.shape)
    dimmed_image = (c*image + (1-c)*black)
    # mask = torch.zeros(image.shape[:2]+(1,))
    # mask[200:300,200:300] = 1
    masked_image = (mask*image + (1-mask)*dimmed_image)
    return masked_image

def plot_masks(image, masks_bool, verbose=False):
    plt.figure()
    plt.imshow(image)

    if verbose:
        iters = tqdm(range(masks_bool.shape[0]))
    else:
        iters = range(masks_bool.shape[0])
    for i in iters:
        mask = masks_bool[i].unsqueeze(-1).float()
        plt.contour(mask[:,:,0], 2, colors='white', linestyles='dashed')

    plt.show()
    