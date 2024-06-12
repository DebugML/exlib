from datasets import load_dataset
from exlib.datasets import massmaps
from exlib.datasets.massmaps import MassMapsConvnetForImageRegression
# Alignment
from exlib.datasets.massmaps import MassMapsAlignment

from exlib.features.massmaps import MassMapsWatershed, MassMapsQuickshift, MassMapsPatch, MassMapsOracle, MassMapsOne


def get_mass_maps_scores(baselines = ['patch', 'quickshift', 'watershed']): # currently we just assume we are running everything, need to update though to be able to specify a baseline to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model
    model = MassMapsConvnetForImageRegression.from_pretrained(massmaps.MODEL_REPO) # BrachioLab/massmaps-conv
    model = model.to(device)
    
    # Load data
    train_dataset = load_dataset(massmaps.DATASET_REPO, split='train') # BrachioLab/massmaps-cosmogrid-100k
    val_dataset = load_dataset(massmaps.DATASET_REPO, split='validation')
    test_dataset = load_dataset(massmaps.DATASET_REPO, split='test')
    train_dataset.set_format('torch', columns=['input', 'label'])
    val_dataset.set_format('torch', columns=['input', 'label'])
    test_dataset.set_format('torch', columns=['input', 'label'])
    
    massmaps_align = MassMapsAlignment()
    
    # Eval
    watershed_baseline = MassMapsWatershed().to(device)
    quickshift_baseline = MassMapsQuickshift().to(device)
    patch_baseline = MassMapsPatch().to(device)
    oracle_baseline = MassMapsOracle().to(device)
    one_baseline = MassMapsOne().to(device)
    
    baselines = {
        'watershed': watershed_baseline,
        'quickshift': quickshift_baseline,
        'patch': patch_baseline,
        'oracle': oracle_baseline,
        'one': one_baseline
    }
    
    batch_size = 16
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    
    model.eval()
    mse_loss_all = 0
    total = 0
    alignment_scores_all = defaultdict(list)
    
    with torch.no_grad():
        for bi, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # if bi % 100 != 0:
            #     continue
            X = batch['input'].to(device)
            y = batch['label'].to(device)
            out = model(X)
            # loss
            loss = F.mse_loss(out, y, reduction='none')
            mse_loss_all = mse_loss_all + loss.sum(0)
            total += X.shape[0]
    
            # baseline
            for name, baseline in baselines.items():
                groups = baseline(X)
    
                # alignment
                alignment_scores = massmaps_align(groups, X)
                alignment_scores_all[name].extend(alignment_scores.flatten(1).cpu().numpy().tolist())
            
            
                
    loss_avg = mse_loss_all / total
    
    print(f'Omega_m loss {loss_avg[0].item():.4f}, sigma_8 loss {loss_avg[1].item():.4f}, avg loss {loss_avg.mean().item():.4f}')

    return alignment_scores_all