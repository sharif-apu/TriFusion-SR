import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import math
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage, structural_similarity as ssim_skimage
import lpips

# =====================
# CONFIGURATION
# =====================
class Config:
    def __init__(self):
        # Dataset paths (pre-split)
        self.data_root = "/home/sharif/Downloads/Praktikumcv/Dataset_Split"
        self.lr_size = 64  # Options: 32, 64, 128
        self.image_size_hr = 256

        # TMFA Model Parameters
        self.encoder_channels = 32 # Channels for initial convolution in modality encoders
        self.reduction_ratio = 16 # Reduction ratio for SE block FC layers in TMFA
        self.conditional_channels = 64 # Output channels of TMFA, input to U-Net as conditioning
        
        # Training
        self.batch_size = 4
        self.num_workers = 4
        self.num_epochs = 1000
        self.learning_rate = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation & Saving
        self.eval_interval = 20
        self.save_interval = 20
        self.model_save_path = f"diffusion_wra_{self.lr_size}to{self.image_size_hr}.pth"#f"diffusion_sr_wsa_uncertain_{self.lr_size}to{self.image_size_hr}.pth"
        self.best_model_path = f"best_wra_{self.lr_size}to{self.image_size_hr}.pth"#f"best_model_wsa_uncertain_{self.lr_size}to{self.image_size_hr}.pth"
        
        # Model architecture
        self.encoder_channels = 32
        self.conditional_channels = 64
        self.base_channels = 64
        self.channel_multipliers = [1, 2, 4, 8]
        self.num_res_blocks = 2
        self.attn_resolutions = [32, 16]
        
        # Diffusion
        self.timesteps = 1000
        self.beta_schedule = 'linear'
        self.start_beta = 0.0001
        self.end_beta = 0.02
        self.loss_type = 'l1'
        
        # Derived
        self.image_channels_hr = 5  # T1, T2, PET(3)
        self.upscale_factor = self.image_size_hr // self.lr_size

cfg = Config()
print(f"Using device: {cfg.device}")
print(f"Training: {cfg.lr_size}x{cfg.lr_size} → {cfg.image_size_hr}x{cfg.image_size_hr}")

# =====================
# DATASET
# =====================
class TrimodalDataset(Dataset):
    def __init__(self, cfg, split='train'):
        """
        split: 'train', 'val', or 'test'
        """
        self.cfg = cfg
        self.split = split
        
        # Paths
        lr_root = os.path.join(cfg.data_root, f"LR_{cfg.lr_size}", split)
        hr_root = os.path.join(cfg.data_root, f"HR_{cfg.image_size_hr}", split)
        
        self.t1_lr = os.path.join(lr_root, "T1-MRI")
        self.t2_lr = os.path.join(lr_root, "T2-MRI")
        self.pet_lr = os.path.join(lr_root, "PET")
        
        self.t1_hr = os.path.join(hr_root, "T1-MRI")
        self.t2_hr = os.path.join(hr_root, "T2-MRI")
        self.pet_hr = os.path.join(hr_root, "PET")
        
        # Get filenames
        self.filenames = sorted([f for f in os.listdir(self.t1_lr) if f.endswith('.png')])
        
        print(f"{split.upper()} set: {len(self.filenames)} samples")
        
        # Transforms
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # Load LR - Force RGB for PET
        t1_lr = self.gray_transform(Image.open(os.path.join(self.t1_lr, fname)))
        t2_lr = self.gray_transform(Image.open(os.path.join(self.t2_lr, fname)))
        pet_lr_img = Image.open(os.path.join(self.pet_lr, fname)).convert('RGB')  # ✅ Force RGB
        pet_lr = self.rgb_transform(pet_lr_img)
        
        # Load HR - Force RGB for PET
        t1_hr = self.gray_transform(Image.open(os.path.join(self.t1_hr, fname)))
        t2_hr = self.gray_transform(Image.open(os.path.join(self.t2_hr, fname)))
        pet_hr_img = Image.open(os.path.join(self.pet_hr, fname)).convert('RGB')  # ✅ Force RGB
        pet_hr = self.rgb_transform(pet_hr_img)
        
        # Fused HR target
        gt_hr = torch.cat([t1_hr, t2_hr, pet_hr], dim=0)  # (5, H, W)
        
        # Normalize to [-1, 1]
        normalize = lambda x: x * 2 - 1
        t1_lr, t2_lr, pet_lr, gt_hr = map(normalize, [t1_lr, t2_lr, pet_lr, gt_hr])
        
        return {
            't1_lr': t1_lr,      # (1, H, W)
            't2_lr': t2_lr,      # (1, H, W)
            'pet_lr': pet_lr,    # (3, H, W)
            'gt_hr': gt_hr,      # (5, H, W)
            'filename': fname
        }

# =====================
# CREATE DATALOADERS
# =====================
print("\n=== Creating Datasets ===")
train_dataset = TrimodalDataset(cfg, 'train')
val_dataset = TrimodalDataset(cfg, 'val')
test_dataset = TrimodalDataset(cfg, 'test')

train_loader = DataLoader(
    train_dataset, 
    batch_size=cfg.batch_size, 
    shuffle=True, 
    num_workers=cfg.num_workers, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=1,  # Batch size 1 for validation
    shuffle=False, 
    num_workers=cfg.num_workers, 
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=1,  # Batch size 1 for testing
    shuffle=False, 
    num_workers=cfg.num_workers, 
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# =====================
# VERIFY DATA LOADING
# =====================
print("\n=== Verifying Data Shapes ===")
sample = next(iter(train_loader))
print(f"t1_lr shape: {sample['t1_lr'].shape}")    # Should be [4, 1, 128, 128]
print(f"t2_lr shape: {sample['t2_lr'].shape}")    # Should be [4, 1, 128, 128]
print(f"pet_lr shape: {sample['pet_lr'].shape}")  # Should be [4, 3, 128, 128] ✅
print(f"gt_hr shape: {sample['gt_hr'].shape}")    # Should be [4, 5, 256, 256]

# =====================
# IMPORT YOUR MODELS
# =====================
# Import all model components from your model.py file
from model import (
    # Wavelet_ASFE_Fusion_Model,
    U_Net,
    DDPM_Scheduler,
    Wavelet_ASFE_Fusion_Model_Simple_Uncertainty,
    Baseline_Simple_Fusion
)

print("\n=== Initializing Models ===")
ddpm_scheduler = DDPM_Scheduler(cfg)
class DiffusionSRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tmfa = Wavelet_ASFE_Fusion_Model_Simple_Uncertainty(cfg)
        #self.tmfa = TMFA_Model(cfg)
        self.unet = U_Net(cfg)

    def forward(self, x_noisy, timesteps, conditional_features=None,
                t1_lr=None, t2_lr=None, pet_lr=None):
        """
        Handles both direct and full-conditioning calls:
        - If conditional_features is provided, uses it directly.
        - Otherwise, computes it from the LR inputs (t1_lr, t2_lr, pet_lr).
        """
        timesteps = timesteps.to(x_noisy.device)

        # Compute conditional features if not provided
        if conditional_features is None:
            if t1_lr is None or t2_lr is None or pet_lr is None:
                raise ValueError("Need either conditional_features or all LR inputs.")
            conditional_features = self.tmfa(t1_lr, t2_lr, pet_lr)

        # 🔥 Upsample conditional features from LR → HR
        if conditional_features.shape[2:] != x_noisy.shape[2:]:
            conditional_features = F.interpolate(
                conditional_features,
                size=x_noisy.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Predict noise with U-Net
        predicted_noise = self.unet(x_noisy, timesteps, conditional_features)
        return predicted_noise
    
diffusion_sr_model = DiffusionSRModel(cfg).to(cfg.device)
print(f"Model parameters: {sum(p.numel() for p in diffusion_sr_model.parameters()):,}")

# =====================
# METRICS
# =====================
def denormalize(tensor):
    """[-1,1] → [0,1]"""
    return ((tensor + 1) / 2).clamp(0, 1)

def calculate_psnr(img1, img2):
    return psnr_skimage(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2, multichannel=False):
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(3, win_size)
    
    if multichannel:
        return ssim_skimage(img1, img2, data_range=1.0, win_size=win_size, channel_axis=2)
    return ssim_skimage(img1, img2, data_range=1.0, win_size=win_size)

lpips_model = lpips.LPIPS(net='alex').to(cfg.device).eval()

def calculate_lpips(img1, img2, multichannel=True):
    def prep(img):
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return (img * 2 - 1).unsqueeze(0).to(cfg.device)
    
    with torch.no_grad():
        return lpips_model(prep(img1), prep(img2)).item()

# =====================
# TRAINING FUNCTIONS
# =====================
def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
    for batch in pbar:
        t1_lr = batch['t1_lr'].to(cfg.device)
        t2_lr = batch['t2_lr'].to(cfg.device)
        pet_lr = batch['pet_lr'].to(cfg.device)
        gt_hr = batch['gt_hr'].to(cfg.device)
        
        # Forward
        timesteps = torch.randint(0, cfg.timesteps, (gt_hr.shape[0],), device=cfg.device).long()
        cond = model.tmfa(t1_lr, t2_lr, pet_lr)
        loss = scheduler.p_losses(model.unet, gt_hr, timesteps, cond)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return np.mean(losses)

def validate(model, loader, scheduler):
    model.eval()
    metrics = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            t1_lr = batch['t1_lr'].to(cfg.device)
            t2_lr = batch['t2_lr'].to(cfg.device)
            pet_lr = batch['pet_lr'].to(cfg.device)
            gt_hr = batch['gt_hr'].to(cfg.device)
            
            cond = model.tmfa(t1_lr, t2_lr, pet_lr)
            sr = scheduler.p_sample_loop(model.unet, gt_hr.shape, cond)
            
            # Convert to numpy
            sr = denormalize(sr[0]).cpu().numpy()
            gt = denormalize(gt_hr[0]).cpu().numpy()
            
            sr_t1, sr_t2, sr_pet = sr[0], sr[1], sr[2:5].transpose(1,2,0)
            gt_t1, gt_t2, gt_pet = gt[0], gt[1], gt[2:5].transpose(1,2,0)
            
            metrics.append({
                'psnr_t1': calculate_psnr(sr_t1, gt_t1),
                'psnr_t2': calculate_psnr(sr_t2, gt_t2),
                'psnr_pet': calculate_psnr(sr_pet, gt_pet),
                'ssim_t1': calculate_ssim(sr_t1, gt_t1, False),
                'ssim_t2': calculate_ssim(sr_t2, gt_t2, False),
                'ssim_pet': calculate_ssim(sr_pet, gt_pet, True),
                'lpips_t1': calculate_lpips(sr_t1, gt_t1, False),
                'lpips_t2': calculate_lpips(sr_t2, gt_t2, False),
                'lpips_pet': calculate_lpips(sr_pet, gt_pet, True),
            })
    
    avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
    return avg_metrics

# =====================
# MAIN TRAINING LOOP
# =====================
print("\n=== Starting Training ===")
optimizer = torch.optim.AdamW(diffusion_sr_model.parameters(), lr=cfg.learning_rate)

best_score = -float('inf')
best_metrics = {}

for epoch in range(cfg.num_epochs):
    # Train
    train_loss = train_epoch(diffusion_sr_model, train_loader, optimizer, ddpm_scheduler, epoch)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
    
    # Validate
    if (epoch + 1) % cfg.eval_interval == 0:
        metrics = validate(diffusion_sr_model, val_loader, ddpm_scheduler)
        
        # Compute score
        avg_psnr = (metrics['psnr_t1'] + metrics['psnr_t2'] + metrics['psnr_pet']) / 3
        avg_ssim = (metrics['ssim_t1'] + metrics['ssim_t2'] + metrics['ssim_pet']) / 3
        avg_lpips = (metrics['lpips_t1'] + metrics['lpips_t2'] + metrics['lpips_pet']) / 3
        score = avg_psnr + avg_ssim - avg_lpips
        
        print(f"Val | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
        
        # Save best
        if score > best_score:
            best_score = score
            best_metrics = {'epoch': epoch+1, 'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips}
            torch.save(diffusion_sr_model.state_dict(), cfg.best_model_path)
            print(f"🏆 New best model saved! Score: {score:.4f}")
    
    # Checkpoint
    if (epoch + 1) % cfg.save_interval == 0:
        torch.save(diffusion_sr_model.state_dict(), cfg.model_save_path)

print("\n✅ Training complete!")
print(f"Best model: Epoch {best_metrics['epoch']} | PSNR: {best_metrics['psnr']:.2f}")

# =====================
# TESTING
# =====================
print("\n==>> Testing best model")
diffusion_sr_model.load_state_dict(torch.load(cfg.best_model_path))
test_metrics = validate(diffusion_sr_model, test_loader, ddpm_scheduler)

print("\n=== FINAL TEST RESULTS ===")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")
