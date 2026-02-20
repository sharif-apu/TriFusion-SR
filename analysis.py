import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

# Import your existing setup
from testing import TestConfig, TrimodalTestDataset, denormalize_img_to_numpy
from model import Wavelet_ASFE_Fusion_Model_Simple_Uncertainty

# --- HELPER FUNCTIONS ---

def robust_normalize(img_data):
    """Normalize using 2nd/98th percentile to fix contrast."""
    flat = img_data.flatten()
    vmin = np.percentile(flat, 2)
    vmax = np.percentile(flat, 98)
    img_norm = np.clip((img_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return img_norm

def remove_padding_artifacts(img, crop_pixels=2):
    """
    Crops outer pixels to remove CNN border artifacts.
    Handles both 2D (H,W) and 3D (H,W,C) images.
    """
    if img.ndim == 2:
        h, w = img.shape
        return img[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels]
    elif img.ndim == 3:
        h, w, c = img.shape
        return img[crop_pixels:h-crop_pixels, crop_pixels:w-crop_pixels, :]
    else:
        return img

def create_professional_overlay(t1_tensor, pet_tensor):
    """Creates the T1 + PET Heatmap Overlay."""
    t1 = denormalize_img_to_numpy(t1_tensor)
    pet = denormalize_img_to_numpy(pet_tensor)
    
    # Normalize
    t1_norm = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)
    pet_norm = (pet - pet.min()) / (pet.max() - pet.min() + 1e-8)
    
    # Enhance T1 Contrast
    t1_norm = np.power(t1_norm, 0.8) 
    
    # Colorize PET
    pet_mask = pet_norm
    pet_mask[pet_mask < 0.15] = 0 
    pet_colored = cv2.applyColorMap((pet_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pet_colored = cv2.cvtColor(pet_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend
    t1_rgb = np.stack([t1_norm]*3, axis=-1)
    alpha = 0.6 * (pet_mask > 0.15)[:, :, None]
    overlay = t1_rgb * (1 - alpha) + pet_colored * alpha
    
    return np.clip(overlay, 0, 1)

def get_feature_maps(cfg, model, batch):
    """Extracts features from all stages."""
    model.eval()
    with torch.no_grad():
        t1 = batch['t1_lr'].to(cfg.device)
        t2 = batch['t2_lr'].to(cfg.device)
        pet = batch['pet_lr'].to(cfg.device)

        # 1. ENCODER
        t1_e = model.t1_encoder(t1)
        t2_e = model.t2_encoder(t2)
        pet_e = model.pet_encoder(pet)
        vis_enc = torch.mean(t1_e, dim=1, keepdim=True)

        # 2. WAVELET
        _, t1_hf = model.t1_wdb(t1_e)
        _, t2_hf = model.t2_wdb(t2_e)
        _, pet_hf = model.pet_wdb(pet_e)
        total_hf = (torch.abs(t1_hf) + torch.abs(t2_hf) + torch.abs(pet_hf))
        vis_wav = torch.mean(total_hf, dim=1, keepdim=True)

        # 3. FUSION
        t1_lf, _ = model.t1_wdb(t1_e)
        t2_lf, _ = model.t2_wdb(t2_e)
        pet_lf, _ = model.pet_wdb(pet_e)
        
        h, w = t1_hf.shape[2], t1_hf.shape[3]
        t1_lf = F.interpolate(t1_lf, size=(h, w))
        t2_lf = F.interpolate(t2_lf, size=(h, w))
        pet_lf = F.interpolate(pet_lf, size=(h, w))
        
        fused = torch.cat([t1_lf, t1_hf, t2_lf, t2_hf, pet_lf, pet_hf], dim=1)
        vis_asfe = torch.mean(torch.abs(fused), dim=1, keepdim=True)

        # 4. REFINEMENT
        _, log_var = model.uncertainty_estimator(fused)
        vis_ref = torch.exp(log_var).mean(dim=1, keepdim=True)

        return vis_enc, vis_wav, vis_asfe, vis_ref

# --- MAIN PLOTTING LOOP ---

def save_all_samples_with_headers(cfg, model, dataset, output_dir):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving individual sample strips to {output_dir}...")
    
    # Professional Headers
    headers = [
        "Trimodal Input\n(T1 Anatomy + PET)", 
        "Encoder Features\n(w/o Wavelet)", 
        "Wavelet Edges\n(High-Freq Extraction)", 
        "Fused ASFE Features\n(w/o Refinement)", 
        "Refined Attention\n(Proposed)"
    ]

    # Loop through ALL samples
    for i, batch in enumerate(tqdm(loader)):
        
        # 1. Prepare Data
        overlay = create_professional_overlay(batch['t1_lr'][0,0].unsqueeze(0), batch['pet_lr'][0,0].unsqueeze(0))
        enc, wav, asfe, ref = get_feature_maps(cfg, model, batch)
        
        # 2. Normalize
        img_enc = robust_normalize(enc[0,0].cpu().numpy())
        img_wav = robust_normalize(wav[0,0].cpu().numpy())
        img_asfe = robust_normalize(asfe[0,0].cpu().numpy())
        img_ref = robust_normalize(ref[0,0].cpu().numpy())

        # 3. Clean Artifacts
        overlay = remove_padding_artifacts(overlay, 2)
        img_enc = remove_padding_artifacts(img_enc, 2)
        img_wav = remove_padding_artifacts(img_wav, 2)
        img_asfe = remove_padding_artifacts(img_asfe, 2)
        img_ref = remove_padding_artifacts(img_ref, 2)

        # 4. Create Figure (1 Row, 5 Cols)
        # Increased figure height slightly to accommodate headers
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
        
        # 5. Plot Columns with Distinct Themes
        axes[0].imshow(overlay)
        axes[1].imshow(img_enc, cmap='viridis')
        axes[2].imshow(img_wav, cmap='magma')   
        axes[3].imshow(img_asfe, cmap='plasma') 
        axes[4].imshow(img_ref, cmap='inferno') 
        
        # 6. Formatting & Headers
        for idx, ax in enumerate(axes):
            ax.set_axis_off() # Remove all ticks/spines
            # Add Header
            ax.set_title(headers[idx], fontsize=13, fontweight='bold', pad=10)

        # 7. Save
        save_name = os.path.join(output_dir, f"Sample_{i:04d}_Evolution.pdf")
        # bbox_inches='tight' ensures headers are not cut off
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig) 

if __name__ == "__main__":
    cfg = TestConfig()
    model = Wavelet_ASFE_Fusion_Model_Simple_Uncertainty(cfg).to(cfg.device)
    
    # Load Weights
    ckpt = torch.load(cfg.best_model_path, map_location=cfg.device)
    state = {k.replace('tmfa.', ''): v for k, v in ckpt.items() if k.startswith('tmfa.')}
    model.load_state_dict(state, strict=False)
    
    dataset = TrimodalTestDataset(cfg, split='test')
    
    # Save folder
    save_folder = os.path.join(cfg.test_output_dir, "Evolution_Strips_With_Headers")
    save_all_samples_with_headers(cfg, model, dataset, save_folder)
