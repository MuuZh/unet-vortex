import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from train import load_muvprocessed_data, gen_mesh_from_boundaries, SavedVortexDataset
import torch

boundaries = np.load(Path('boundaries7parcellation.npy'))
brainmesh = gen_mesh_from_boundaries(boundaries)
brainmesh = np.where(brainmesh == 0, np.nan, brainmesh)

datapath = 'prepre/validate_data.pkl'

test_data = load_muvprocessed_data(datapath, brainmesh)
test_dataset = test_data

predictions = torch.load("predictions/predictions.pt")

num_samples = 3
random_indices = np.random.choice(len(predictions), num_samples, replace=False)
fig, axes = plt.subplots(num_samples, 4, figsize=(25, 5 * num_samples))

for i, idx in enumerate(random_indices):
    sample = test_dataset[idx]
    vx = sample['velocity'][0].numpy()
    vy = sample['velocity'][1].numpy()
    vorticity = sample['vorticity'][0].numpy()
    ivd = sample['ivd'][0].numpy()
    label = sample['label'][1].numpy()

    pred = predictions[idx][1].numpy()
    pred = (pred > 0.5).astype(float)

    im1 = axes[i, 0].imshow(label * brainmesh, cmap='gray')
    axes[i, 0].quiver(vx * brainmesh, vy * brainmesh, scale=550, color='red')
    axes[i, 0].set_title(f'Sample {idx} - Ground Truth')
    plt.colorbar(im1, ax=axes[i, 0])
    axes[i, 0].set_aspect('equal')
    axes[i, 0].set_xlim(0, 251)
    axes[i, 0].set_ylim(0, 175)

    im2 = axes[i, 1].imshow(pred * brainmesh, cmap='gray')
    axes[i, 1].quiver(vx * brainmesh, vy * brainmesh, scale=550, color='red')
    axes[i, 1].set_title(f'Sample {idx} - Prediction')
    plt.colorbar(im2, ax=axes[i, 1])
    axes[i, 1].set_aspect('equal')
    axes[i, 1].set_xlim(0, 251)
    axes[i, 1].set_ylim(0, 175)

    im3 = axes[i, 2].imshow(vorticity * brainmesh, cmap='viridis')
    axes[i, 2].set_title(f'Sample {idx} - Vorticity')
    plt.colorbar(im3, ax=axes[i, 2])
    axes[i, 2].set_aspect('equal')
    axes[i, 2].set_xlim(0, 251)
    axes[i, 2].set_ylim(0, 175)

    im4 = axes[i, 3].imshow(ivd * brainmesh, cmap='plasma')
    axes[i, 3].set_title(f'Sample {idx} - IVD')
    plt.colorbar(im4, ax=axes[i, 3])
    axes[i, 3].set_aspect('equal')
    axes[i, 3].set_xlim(0, 251)
    axes[i, 3].set_ylim(0, 175)

plt.tight_layout()
plt.show()