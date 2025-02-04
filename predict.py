import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_fill_holes

from train import MVUNet, SavedVortexDataset, DoubleConv, load_muvprocessed_data, gen_mesh_from_boundaries

def predict(model, test_loader, device, output_dir="predictions"):

    os.makedirs(output_dir, exist_ok=True)

    all_predictions = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            velocity = batch['velocity'].to(device)
            vorticity = batch['vorticity'].to(device)
            ivd = batch['ivd'].to(device)

            outputs = model(velocity, vorticity, ivd)
            probs = torch.sigmoid(outputs)
            all_predictions.append(probs.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)

    torch.save(all_predictions, os.path.join(output_dir, "predictions.pt"))
    print(f"Predictions saved to {output_dir}/predictions.pt")

if __name__ == "__main__":
    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MVUNet().to(device)
    checkpoint = torch.load("save_weights/best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    boundaries = np.load(Path('boundaries7parcellation.npy'))
    brainmesh = gen_mesh_from_boundaries(boundaries)
    datapath = 'prepre/validate_data.pkl'
    test_data = load_muvprocessed_data(datapath, brainmesh)
    print('size of test_data:', len(test_data))

    test_loader = DataLoader(
        test_data,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    predict(model, test_loader, device)