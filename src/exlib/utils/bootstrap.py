import torch

def bootstrap(scores, n_samples=5, seed=42):
    scores_samples = []
    means = []
    torch.manual_seed(seed)

    for _ in range(n_samples):
        scores_new = torch.zeros(scores.shape)
        for i in range(len(scores_new)):
            # print(i)
            index = torch.randperm(len(scores_new))[0]
            scores_new[i] = scores[index]

        means.append(torch.mean(scores_new))
        scores_samples.append(scores_new)
    return torch.std(torch.stack(means))