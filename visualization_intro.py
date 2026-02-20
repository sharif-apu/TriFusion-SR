import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ============================================================
#  ASSUMPTION: You have these files in your local directory
# ============================================================
from model import *
from testing import TestConfig, TrimodalTestDataset

# ============================================================
# 🎛️ GLOBAL PLOT CONFIGURATION
# ============================================================

# 1. TOTAL CANVAS SIZE (Inches)
FIG_WIDTH = 8.   
FIG_HEIGHT = 10. 

# 2. ROW HEIGHT RATIOS
IMG_ROW_HEIGHT = 1.0    # Height of T1, T2, PET rows
GRAPH_ROW_HEIGHT = 1.2  # Height of the bottom graph

# 3. GRAPH SPECIFIC WIDTH CONTROL (New!)
GRAPH_WIDTH_SCALE = .95   # 1.0 = Matches image width. >1.0 = Wider. <1.0 = Narrower.
GRAPH_LEFT_SHIFT = 0.01    # (-) moves Left, (+) moves Right. Try -0.05 to fix label alignment.

# 4. SPACING & MARGINS
W_SPACE = 0.01   # Gap between columns
H_SPACE = 0.01   # Gap between rows
LEFT_MARGIN = 0.0
RIGHT_MARGIN = 0.95
TOP_MARGIN = 0.95
BOTTOM_MARGIN = 0.15

# 5. LEGEND POSITION
LEGEND_POS = (0.5, -0.35) 

# ============================================================
#               FREQUENCY ANALYSIS FUNCTIONS
# ============================================================

def get_fft_spectrum(residual_tensor):
    img_np = residual_tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.mean(img_np, axis=0)
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

def get_radial_profile(spectrum):
    y, x = np.indices(spectrum.shape)
    center = np.array(spectrum.shape) / 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

# ============================================================
#               PLOTTING HELPER
# ============================================================

def show_img(ax, img, title=None, ylabel=None, cmap='gray', is_spectrum=False):
    if isinstance(img, torch.Tensor):
        img = (img.squeeze().cpu().numpy() + 1) / 2
    elif isinstance(img, np.ndarray):
        pass 

    if img.ndim == 3: 
        img = img.transpose(1, 2, 0)
    
    aspect = 'auto' 
    
    if is_spectrum:
        ax.imshow(img, cmap=cmap, aspect=aspect)
    else:
        ax.imshow(img.clip(0, 1), cmap=cmap, aspect=aspect)
    
    if title: 
        ax.set_title(title, fontsize=9, pad=4)
    if ylabel: 
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', labelpad=4)
        
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ============================================================
#               MAIN PROCESSING LOOPS
# ============================================================

def run_miccai_justification_analysis(cfg):
    print("\n" + "="*70)
    print("   MICCAI MOTIVATION ANALYSIS")
    print("="*70)
    
    save_dir = os.path.join(cfg.test_output_dir, "miccai_frequency_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = TrimodalTestDataset(cfg, split='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for idx, batch in enumerate(tqdm(loader, desc="Analyzing Samples")):
        filename = batch['filename'][0]
        
        t1_lr = batch['t1_lr'].to(cfg.device)
        t2_lr = batch['t2_lr'].to(cfg.device)
        pet_lr = batch['pet_lr'].to(cfg.device)
        gt_hr = batch['gt_hr'].to(cfg.device)
        
        t1_gt = gt_hr[:, 0:1]
        t2_gt = gt_hr[:, 1:2]
        pet_gt = gt_hr[:, 2:5]
        
        t1_lr_up = F.interpolate(t1_lr, size=(256, 256), mode='bicubic', align_corners=False)
        t2_lr_up = F.interpolate(t2_lr, size=(256, 256), mode='bicubic', align_corners=False)
        pet_lr_up = F.interpolate(pet_lr, size=(256, 256), mode='bicubic', align_corners=False)
        
        res_t1 = t1_gt - t1_lr_up
        res_t2 = t2_gt - t2_lr_up
        res_pet = pet_gt - pet_lr_up
        
        spec_t1 = get_fft_spectrum(res_t1[0])
        spec_t2 = get_fft_spectrum(res_t2[0])
        spec_pet = get_fft_spectrum(res_pet[0])
        
        prof_t1 = get_radial_profile(spec_t1)
        prof_t2 = get_radial_profile(spec_t2)
        prof_pet = get_radial_profile(spec_pet)
        
        # --- PLOTTING ---
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        
        hr = [IMG_ROW_HEIGHT, IMG_ROW_HEIGHT, IMG_ROW_HEIGHT, GRAPH_ROW_HEIGHT]
        
        gs = fig.add_gridspec(4, 3, 
                             height_ratios=hr,
                             width_ratios=[1, 1, 1],
                             hspace=H_SPACE, wspace=W_SPACE, 
                             left=LEFT_MARGIN, right=RIGHT_MARGIN,
                             top=TOP_MARGIN, bottom=BOTTOM_MARGIN)

        # Row 1
        show_img(fig.add_subplot(gs[0, 0]), t1_lr_up, "Input", ylabel="T1")
        show_img(fig.add_subplot(gs[0, 1]), t1_gt, "Ground Truth")
        show_img(fig.add_subplot(gs[0, 2]), spec_t1, "Spectrum", cmap='inferno', is_spectrum=True)
        
        # Row 2
        show_img(fig.add_subplot(gs[1, 0]), t2_lr_up, ylabel="T2")
        show_img(fig.add_subplot(gs[1, 1]), t2_gt)
        show_img(fig.add_subplot(gs[1, 2]), spec_t2, cmap='inferno', is_spectrum=True)
        
        # Row 3
        show_img(fig.add_subplot(gs[2, 0]), pet_lr_up, ylabel="PET", cmap=None)
        show_img(fig.add_subplot(gs[2, 1]), pet_gt, cmap=None)
        show_img(fig.add_subplot(gs[2, 2]), spec_pet, cmap='inferno', is_spectrum=True)
        
        # Row 4 (Graph)
        ax_graph = fig.add_subplot(gs[3, :])
        
        # --- APPLY GRAPH WIDTH CONTROL ---
        pos = ax_graph.get_position()
        new_width = pos.width * GRAPH_WIDTH_SCALE
        # Center the scaling, then apply shift
        new_left = pos.x0 + (pos.width - new_width) / 2 + GRAPH_LEFT_SHIFT
        ax_graph.set_position([new_left, pos.y0, new_width, pos.height])
        # ---------------------------------
        
        max_freq = len(prof_t1) // 2
        x = np.linspace(0, 1, max_freq)
        
        ax_graph.plot(x, prof_t1[:max_freq], label='T1', color='#2E86AB', linewidth=2)
        ax_graph.plot(x, prof_t2[:max_freq], label='T2', color='#A23B72', linewidth=2)
        ax_graph.plot(x, prof_pet[:max_freq], label='PET', color='#F18F01', linewidth=2)
        
        split_point = 0.5
        ax_graph.axvline(x=split_point, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_graph.axvspan(0, split_point, alpha=0.06, color='blue')
        ax_graph.axvspan(split_point, 1.0, alpha=0.06, color='red')
        
        y_max = max(prof_t1.max(), prof_t2.max(), prof_pet.max())
        ax_graph.text(split_point/2, y_max*0.96, "Low Freq", ha='center', fontsize=8, fontweight='bold', color='navy')
        ax_graph.text(split_point + (1-split_point)/2, y_max*0.96, "High Freq", ha='center', fontsize=8, fontweight='bold', color='darkred')
        
        ax_graph.set_xlabel("Normalized Frequency", fontsize=9, labelpad=2)
        ax_graph.set_ylabel("Log Magnitude", fontsize=9, labelpad=2)
        ax_graph.tick_params(labelsize=8)
        ax_graph.grid(True, linestyle=':', alpha=0.4)
        
        # ax_graph.legend(loc='upper center', 
        #                bbox_to_anchor=LEGEND_POS, 
        #                ncol=3, 
        #                frameon=False, 
        #                fontsize=9)
        ax_graph.legend(fontsize=7, loc='lower center', ncol=3)

        plt.savefig(os.path.join(save_dir, f"miccai_{filename.replace(".png", ".pdf")}"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"✅ Analysis complete. Images saved to {save_dir}")

def run_average_frequency_analysis(cfg):
    print("\n" + "="*70)
    print("   MICCAI: AVERAGE FREQUENCY DISTRIBUTION ANALYSIS")
    print("="*70)
    
    dataset = TrimodalTestDataset(cfg, split='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    all_profiles_t1 = []
    all_profiles_t2 = []
    all_profiles_pet = []
    
    rep_idx = 5
    rep_data = {} 

    print("⏳ Accumulating frequency data...")
    
    for idx, batch in enumerate(tqdm(loader)):
        t1_lr = batch['t1_lr'].to(cfg.device)
        t2_lr = batch['t2_lr'].to(cfg.device)
        pet_lr = batch['pet_lr'].to(cfg.device)
        gt_hr = batch['gt_hr'].to(cfg.device)
        
        t1_lr_up = F.interpolate(t1_lr, size=(256, 256), mode='bicubic', align_corners=False)
        t2_lr_up = F.interpolate(t2_lr, size=(256, 256), mode='bicubic', align_corners=False)
        pet_lr_up = F.interpolate(pet_lr, size=(256, 256), mode='bicubic', align_corners=False)
        
        res_t1 = gt_hr[:, 0:1] - t1_lr_up
        res_t2 = gt_hr[:, 1:2] - t2_lr_up
        res_pet = gt_hr[:, 2:5] - pet_lr_up
        
        spec_t1 = get_fft_spectrum(res_t1[0])
        spec_t2 = get_fft_spectrum(res_t2[0])
        spec_pet = get_fft_spectrum(res_pet[0])
        
        prof_t1 = get_radial_profile(spec_t1)
        prof_t2 = get_radial_profile(spec_t2)
        prof_pet = get_radial_profile(spec_pet)
        
        all_profiles_t1.append(prof_t1)
        all_profiles_t2.append(prof_t2)
        all_profiles_pet.append(prof_pet)
        
        if idx == rep_idx:
            rep_data = {
                't1_in': t1_lr_up, 't1_gt': gt_hr[:, 0:1], 't1_spec': spec_t1,
                't2_in': t2_lr_up, 't2_gt': gt_hr[:, 1:2], 't2_spec': spec_t2,
                'pet_in': pet_lr_up, 'pet_gt': gt_hr[:, 2:5], 'pet_spec': spec_pet,
                'filename': batch['filename'][0]
            }

    avg_t1 = np.mean(np.array(all_profiles_t1), axis=0)
    avg_t2 = np.mean(np.array(all_profiles_t2), axis=0)
    avg_pet = np.mean(np.array(all_profiles_pet), axis=0)
    
    print("✅ Generating final plot...")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    hr = [IMG_ROW_HEIGHT, IMG_ROW_HEIGHT, IMG_ROW_HEIGHT, GRAPH_ROW_HEIGHT]
    
    gs = fig.add_gridspec(4, 3, 
                         height_ratios=hr,
                         width_ratios=[1, 1, 1],
                         hspace=H_SPACE, wspace=W_SPACE, 
                         left=LEFT_MARGIN, right=RIGHT_MARGIN,
                         top=TOP_MARGIN, bottom=BOTTOM_MARGIN)
    
    # Row 1
    show_img(fig.add_subplot(gs[0, 0]), rep_data['t1_in'], "Input", ylabel="-MR-T1")
    show_img(fig.add_subplot(gs[0, 1]), rep_data['t1_gt'], "Ground Truth")
    show_img(fig.add_subplot(gs[0, 2]), rep_data['t1_spec'], "Spectrum", cmap='inferno', is_spectrum=True)
    
    # Row 2
    show_img(fig.add_subplot(gs[1, 0]), rep_data['t2_in'], ylabel="MR-T2")
    show_img(fig.add_subplot(gs[1, 1]), rep_data['t2_gt'])
    show_img(fig.add_subplot(gs[1, 2]), rep_data['t2_spec'], cmap='inferno', is_spectrum=True)
    
    # Row 3
    show_img(fig.add_subplot(gs[2, 0]), rep_data['pet_in'], ylabel="SPECT", cmap=None)
    show_img(fig.add_subplot(gs[2, 1]), rep_data['pet_gt'], cmap=None)
    show_img(fig.add_subplot(gs[2, 2]), rep_data['pet_spec'], cmap='inferno', is_spectrum=True)
    
    # Row 4 (Graph)
    ax_graph = fig.add_subplot(gs[3, :])
    
    # --- APPLY GRAPH WIDTH CONTROL ---
    pos = ax_graph.get_position()
    new_width = pos.width * GRAPH_WIDTH_SCALE
    # Center the scaling, then apply shift
    new_left = pos.x0 + (pos.width - new_width) / 2 + GRAPH_LEFT_SHIFT
    ax_graph.set_position([new_left, pos.y0, new_width, pos.height])
    # ---------------------------------
    
    max_freq = len(avg_t1) // 2
    x = np.linspace(0, 1, max_freq)
    
    ax_graph.plot(x, avg_t1[:max_freq], label='MR-T1', color='#2E86AB', linewidth=2.5)
    ax_graph.plot(x, avg_t2[:max_freq], label='MR-T2', color='#A23B72', linewidth=2.5)
    ax_graph.plot(x, avg_pet[:max_freq], label='SPECT', color='#F18F01', linewidth=2.5)
    
    split_point = 0.5
    ax_graph.axvline(x=split_point, color='gray', linestyle='--', linewidth=1.2)
    ax_graph.axvspan(0, split_point, alpha=0.06, color='blue')
    ax_graph.axvspan(split_point, 1.0, alpha=0.06, color='red')
    
    y_max = max(avg_t1.max(), avg_t2.max(), avg_pet.max())
    ax_graph.text(split_point/2, y_max*0.96, "Low Frequency", ha='center', fontsize=8, fontweight='bold', color='navy')
    ax_graph.text(split_point + (1-split_point)/2, y_max*0.96, "High Frequency", ha='center', fontsize=8, fontweight='bold', color='darkred')
    
    ax_graph.set_xlabel("Normalized Frequency", fontsize=9, labelpad=2)
    ax_graph.set_ylabel("Mean Log Amplitude", fontsize=9, labelpad=2)
    ax_graph.tick_params(labelsize=8)
    ax_graph.grid(True, linestyle=':', alpha=0.4)
    
    # ax_graph.legend(loc='upper center', 
    #                bbox_to_anchor=LEGEND_POS, 
    #                ncol=3, 
    #                frameon=False, 
    #                fontsize=9)
    ax_graph.legend(fontsize=7, loc='lower center', ncol=3)
    
    save_path = os.path.join(cfg.test_output_dir, "FINAL_Average_Frequency_Analysis.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Final plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    cfg = TestConfig()
    
    # Run the analysis you need:
    run_miccai_justification_analysis(cfg)
    run_average_frequency_analysis(cfg)
