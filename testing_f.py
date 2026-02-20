import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import csv
import cv2

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import mean_squared_error as mse_skimage
from skimage.filters import sobel
import lpips

# ============================================================
#                    CONFIGURATION
# ============================================================

class TestConfig:
    def __init__(self):
        # Data Paths
        self.data_root_hr = "/home/sharif/Downloads/Praktikumcv/Dataset_Split/HR_256"
        self.data_root_lr = "/home/sharif/Downloads/Praktikumcv/Dataset_Split/LR_64"
        self.t1_dir = "T1-MRI"
        self.t2_dir = "T2-MRI"
        self.pet_dir = "PET"
        
        # Image Dimensions
        self.image_size_hr = 256
        self.image_size_lr = 64
        self.image_channels_hr = 5
        
        # Model Architecture Parameters
        self.encoder_channels = 32
        self.reduction_ratio = 16
        self.conditional_channels = 64
        self.base_channels = 64
        self.channel_multipliers = [1, 2, 4, 8]
        self.num_res_blocks = 2
        self.attn_resolutions = [32, 16]
        
        # Diffusion Parameters
        self.timesteps = 1000
        self.beta_schedule = 'linear'
        self.start_beta = 0.0001
        self.end_beta = 0.02
        
        # Testing Settings
        self.test_split = 'test'
        self.batch_size = 1
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model Path
        self.best_model_path = "/home/sharif/Downloads/Praktikumcv/Code/best_baseline_64to256.pth"
        
        # Output Paths
        self.test_output_dir = "./test_results" + "_" + self.best_model_path.split("/")[-1].split(".")[0]
        self.fused_images_dir = os.path.join(self.test_output_dir, "fused_images")
        self.fused_sr_dir = os.path.join(self.fused_images_dir, "SR")
        self.fused_gt_dir = os.path.join(self.fused_images_dir, "GT")
        self.fused_lr_dir = os.path.join(self.fused_images_dir, "LR")
        
        # Results files
        self.metrics_csv = os.path.join(self.test_output_dir, "per_sample_metrics.csv")
        self.fusion_metrics_csv = os.path.join(self.test_output_dir, "fusion_metrics.csv")
        self.summary_json = os.path.join(self.test_output_dir, "test_summary.json")
        self.summary_txt = os.path.join(self.test_output_dir, "test_summary.txt")
        
        # Create directories
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.fused_sr_dir, exist_ok=True)
        os.makedirs(self.fused_gt_dir, exist_ok=True)
        os.makedirs(self.fused_lr_dir, exist_ok=True)

# ============================================================
#                    DATASET CLASS
# ============================================================

class TrimodalTestDataset(Dataset):
    def __init__(self, cfg, split='test'):
        self.cfg = cfg
        self.split = split
        
        # Paths
        self.t1_hr_path = os.path.join(cfg.data_root_hr, split, cfg.t1_dir)
        self.t2_hr_path = os.path.join(cfg.data_root_hr, split, cfg.t2_dir)
        self.pet_hr_path = os.path.join(cfg.data_root_hr, split, cfg.pet_dir)
        self.t1_lr_path = os.path.join(cfg.data_root_lr, split, cfg.t1_dir)
        self.t2_lr_path = os.path.join(cfg.data_root_lr, split, cfg.t2_dir)
        self.pet_lr_path = os.path.join(cfg.data_root_lr, split, cfg.pet_dir)
        
        # Verify paths
        for path in [self.t1_hr_path, self.t2_hr_path, self.pet_hr_path,
                     self.t1_lr_path, self.t2_lr_path, self.pet_lr_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")
        
        self.filenames = sorted([f for f in os.listdir(self.t1_hr_path) if f.endswith('.png')])
        if not self.filenames:
            raise RuntimeError(f"No .png files found in {self.t1_hr_path}")
        
        print(f"Loaded {split} split: {len(self.filenames)} samples")
        
        # Transforms
        self.grayscale_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.rgb_transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load HR images
        t1_hr = self.grayscale_transform(Image.open(os.path.join(self.t1_hr_path, filename)).convert('L'))
        t2_hr = self.grayscale_transform(Image.open(os.path.join(self.t2_hr_path, filename)).convert('L'))
        pet_hr = self.rgb_transform(Image.open(os.path.join(self.pet_hr_path, filename)).convert('RGB'))
        
        # Load LR images
        t1_lr = self.grayscale_transform(Image.open(os.path.join(self.t1_lr_path, filename)).convert('L'))
        t2_lr = self.grayscale_transform(Image.open(os.path.join(self.t2_lr_path, filename)).convert('L'))
        pet_lr = self.rgb_transform(Image.open(os.path.join(self.pet_lr_path, filename)).convert('RGB'))
        
        # Fused HR ground truth
        gt_hr_fused = torch.cat([t1_hr, t2_hr, pet_hr], dim=0)
        
        # Normalize to [-1, 1]
        normalize = lambda x: (x * 2) - 1
        t1_lr, t2_lr, pet_lr, gt_hr_fused = map(normalize, [t1_lr, t2_lr, pet_lr, gt_hr_fused])
        
        return {
            't1_lr': t1_lr,
            't2_lr': t2_lr,
            'pet_lr': pet_lr,
            'gt_hr': gt_hr_fused,
            'filename': filename
        }

# ============================================================
#                    METRICS FUNCTIONS
# ============================================================

def denormalize_img_to_numpy(tensor):
    """Convert tensor from [-1,1] to [0,1] numpy array"""
    img_01 = ((tensor + 1) / 2).clamp(0, 1).cpu()
    if img_01.ndim == 3:
        if img_01.shape[0] == 1:
            return img_01[0].numpy()
        else:
            return img_01.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported tensor dimensions: {img_01.ndim}")

def to_python_type(value):
    """Convert numpy/torch types to Python native types for JSON serialization"""
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    else:
        return value

def calculate_mse(img1, img2):
    """Mean Squared Error"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    return float(mse_skimage(img1.astype(np.float64), img2.astype(np.float64)))

def calculate_mae(img1, img2):
    """Mean Absolute Error"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))

def calculate_rmse(img1, img2):
    """Root Mean Squared Error"""
    return float(np.sqrt(calculate_mse(img1, img2)))

def calculate_psnr(img1, img2, data_range=1.0):
    """Peak Signal-to-Noise Ratio"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    return float(psnr_skimage(img1.astype(np.float64), img2.astype(np.float64), data_range=data_range))

def calculate_ssim(img1, img2, data_range=1.0, multichannel=False):
    """Structural Similarity Index"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    min_spatial_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_spatial_dim)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    
    if multichannel:
        return float(ssim_skimage(img1.astype(np.float64), img2.astype(np.float64), 
                           data_range=data_range, win_size=win_size, channel_axis=2))
    else:
        return float(ssim_skimage(img1.astype(np.float64), img2.astype(np.float64), 
                           data_range=data_range, win_size=win_size))

def calculate_ag(img):
    """Average Gradient - measures image sharpness"""
    if img.ndim == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img
    
    gx = sobel(img_gray, axis=0, mode='constant')
    gy = sobel(img_gray, axis=1, mode='constant')
    ag = np.mean(np.sqrt(gx**2 + gy**2))
    
    return float(ag)

def calculate_vif(img1, img2, sigma_nsq=2):
    """Visual Information Fidelity"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    # Convert to grayscale if needed
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2)
    
    num_scales = 4
    eps = 1e-10
    vif_val = 0
    
    for scale in range(num_scales):
        if scale > 0:
            img1 = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
            img2 = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2))
        
        # Calculate local statistics
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        sigma1_sq = cv2.GaussianBlur(img1*img1, (11, 11), 1.5) - mu1*mu1
        sigma2_sq = cv2.GaussianBlur(img2*img2, (11, 11), 1.5) - mu2*mu2
        sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1*mu2
        
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        sv_sq = np.maximum(sv_sq, eps)
        
        num = np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den = np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        
        if den != 0:
            vif_val += num / den
    
    return float(max(0, vif_val / num_scales))

def calculate_lpips(img1, img2, lpips_model):
    """Learned Perceptual Image Patch Similarity"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    
    def preprocess(img):
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img.unsqueeze(0)
    
    t1 = preprocess(img1)
    t2 = preprocess(img2)
    
    device = next(lpips_model.parameters()).device
    t1, t2 = t1.to(device), t2.to(device)
    
    with torch.no_grad():
        score = lpips_model(t1, t2)
    
    return float(score.item())

def calculate_all_metrics(img1, img2, lpips_model, is_multichannel=False):
    """Calculate all metrics at once"""
    metrics = {}
    
    # Basic metrics
    metrics['mse'] = calculate_mse(img1, img2)
    metrics['mae'] = calculate_mae(img1, img2)
    metrics['rmse'] = calculate_rmse(img1, img2)
    metrics['psnr'] = calculate_psnr(img1, img2)
    metrics['ssim'] = calculate_ssim(img1, img2, multichannel=is_multichannel)
    
    # Advanced metrics
    try:
        metrics['vif'] = calculate_vif(img1, img2)
    except Exception as e:
        print(f"Warning: VIF calculation failed: {e}")
        metrics['vif'] = 0.0
    
    metrics['ag_pred'] = calculate_ag(img1)
    metrics['ag_gt'] = calculate_ag(img2)
    metrics['lpips'] = calculate_lpips(img1, img2, lpips_model)
    
    return metrics

def create_fusion_visualization(t1_gray, t2_gray, pet_rgb, t1_weight=0.7, pet_alpha=0.6):
    """Create fused visualization from T1, T2, and PET images"""
    # Ensure grayscale inputs are 2D
    if t1_gray.ndim == 3 and t1_gray.shape[2] == 1:
        t1_gray = t1_gray[:,:,0]
    if t2_gray.ndim == 3 and t2_gray.shape[2] == 1:
        t2_gray = t2_gray[:,:,0]
    
    if not (pet_rgb.ndim == 3 and pet_rgb.shape[2] == 3):
        raise ValueError("PET must be 3-channel RGB")
    
    # Fuse T1 and T2 into grayscale MRI
    mri_gray = (t1_gray * t1_weight + t2_gray * (1 - t1_weight)).clip(0, 1)
    
    # Convert to RGB
    mri_rgb = np.stack([mri_gray, mri_gray, mri_gray], axis=-1)
    
    # Convert to BGR for OpenCV
    mri_bgr = (mri_rgb * 255).astype(np.uint8)[:, :, ::-1]
    pet_bgr = (pet_rgb * 255).astype(np.uint8)[:, :, ::-1]
    
    # Blend MRI and PET
    fused_bgr = cv2.addWeighted(mri_bgr, 1 - pet_alpha, pet_bgr, pet_alpha, 0)
    
    # Convert back to RGB
    fused_rgb = fused_bgr[:, :, ::-1].astype(np.float32) / 255.0
    
    return fused_rgb

def save_fused_image(fused_img, save_path):
    """Save fused image as PNG"""
    fused_uint8 = (fused_img * 255).astype(np.uint8)
    Image.fromarray(fused_uint8).save(save_path)

# ============================================================
#                    MAIN TESTING FUNCTION
# ============================================================

def test_model(cfg):
    print("\\n" + "="*70)
    print(" "*15 + "DIFFUSION SR MODEL - TESTING")
    print("="*70)
    print(f"Device:      {cfg.device}")
    print(f"Model:       {cfg.best_model_path}")
    print(f"Test Split:  {cfg.test_split}")
    print(f"Output Dir:  {cfg.test_output_dir}")
    print("="*70 + "\\n")
    
    # Load model components
    from model import DDPM_Scheduler, Wavelet__Rectified_ASFE_Fusion_Model, U_Net
    class DiffusionSRModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.tmfa = Wavelet__Rectified_ASFE_Fusion_Model(cfg)
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
            
    # Load dataset
    print("📂 Loading test dataset...")
    test_dataset = TrimodalTestDataset(cfg, split=cfg.test_split)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    print(f"✅ Loaded {len(test_dataset)} test samples\\n")
    #test_dataset = test_dataset[:2]
    # Load model
    print("🔧 Loading model...")
    model = DiffusionSRModel(cfg).to(cfg.device)
    
    if not os.path.exists(cfg.best_model_path):
        raise FileNotFoundError(f"❌ Model not found: {cfg.best_model_path}")
    
    checkpoint = torch.load(cfg.best_model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Model loaded successfully\\n")
    
    # Initialize scheduler and LPIPS
    scheduler = DDPM_Scheduler(cfg)
    lpips_model = lpips.LPIPS(net='alex').to(cfg.device)
    lpips_model.eval()
    
    # Storage for metrics
    all_metrics = []
    all_fusion_metrics = []
    
    print("🧪 Starting testing...")
    print("="*70 + "\\n")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing", ncols=80)):
            filename = batch['filename'][0]
            
            # Move to device
            t1_lr = batch['t1_lr'].to(cfg.device)
            t2_lr = batch['t2_lr'].to(cfg.device)
            pet_lr = batch['pet_lr'].to(cfg.device)
            gt_hr = batch['gt_hr'].to(cfg.device)
            
            # Generate SR image
            cond = model.tmfa(t1_lr, t2_lr, pet_lr)
            generated = scheduler.p_sample_loop(
                model=model.unet,
                shape=gt_hr.shape,
                conditional_features=cond
            )
            
            # Extract modalities
            sr = generated[0]
            gt = gt_hr[0]
            
            # SR outputs
            sr_t1 = denormalize_img_to_numpy(sr[0].unsqueeze(0))
            sr_t2 = denormalize_img_to_numpy(sr[1].unsqueeze(0))
            sr_pet = denormalize_img_to_numpy(sr[2:5])
            
            # GT outputs
            gt_t1 = denormalize_img_to_numpy(gt[0].unsqueeze(0))
            gt_t2 = denormalize_img_to_numpy(gt[1].unsqueeze(0))
            gt_pet = denormalize_img_to_numpy(gt[2:5])
            
            # LR outputs
            lr_t1 = denormalize_img_to_numpy(t1_lr[0, 0].unsqueeze(0))
            lr_t2 = denormalize_img_to_numpy(t2_lr[0, 0].unsqueeze(0))
            lr_pet = denormalize_img_to_numpy(pet_lr[0])
            
            # Calculate per-modality metrics
            metrics_t1 = calculate_all_metrics(sr_t1, gt_t1, lpips_model, False)
            metrics_t2 = calculate_all_metrics(sr_t2, gt_t2, lpips_model, False)
            metrics_pet = calculate_all_metrics(sr_pet, gt_pet, lpips_model, True)
            
            # Create fused images
            sr_fused = create_fusion_visualization(sr_t1, sr_t2, sr_pet, 0.7, 0.6)
            gt_fused = create_fusion_visualization(gt_t1, gt_t2, gt_pet, 0.7, 0.6)
            lr_fused = create_fusion_visualization(lr_t1, lr_t2, lr_pet, 0.7, 0.6)
            
            # Save fused images
            save_fused_image(sr_fused, os.path.join(cfg.fused_sr_dir, filename))
            save_fused_image(gt_fused, os.path.join(cfg.fused_gt_dir, filename))
            save_fused_image(lr_fused, os.path.join(cfg.fused_lr_dir, filename))
            
            # Calculate fusion metrics
            fusion_metrics = calculate_all_metrics(sr_fused, gt_fused, lpips_model, True)
            
            # Store per-sample metrics (convert all to Python native types)
            sample_metrics = {
                'idx': int(idx + 1),
                'filename': filename,
                # T1 metrics
                'mse_t1': to_python_type(metrics_t1['mse']),
                'mae_t1': to_python_type(metrics_t1['mae']),
                'rmse_t1': to_python_type(metrics_t1['rmse']),
                'psnr_t1': to_python_type(metrics_t1['psnr']),
                'ssim_t1': to_python_type(metrics_t1['ssim']),
                'vif_t1': to_python_type(metrics_t1['vif']),
                'lpips_t1': to_python_type(metrics_t1['lpips']),
                'ag_t1': to_python_type(metrics_t1['ag_pred']),
                # T2 metrics
                'mse_t2': to_python_type(metrics_t2['mse']),
                'mae_t2': to_python_type(metrics_t2['mae']),
                'rmse_t2': to_python_type(metrics_t2['rmse']),
                'psnr_t2': to_python_type(metrics_t2['psnr']),
                'ssim_t2': to_python_type(metrics_t2['ssim']),
                'vif_t2': to_python_type(metrics_t2['vif']),
                'lpips_t2': to_python_type(metrics_t2['lpips']),
                'ag_t2': to_python_type(metrics_t2['ag_pred']),
                # PET metrics
                'mse_pet': to_python_type(metrics_pet['mse']),
                'mae_pet': to_python_type(metrics_pet['mae']),
                'rmse_pet': to_python_type(metrics_pet['rmse']),
                'psnr_pet': to_python_type(metrics_pet['psnr']),
                'ssim_pet': to_python_type(metrics_pet['ssim']),
                'vif_pet': to_python_type(metrics_pet['vif']),
                'lpips_pet': to_python_type(metrics_pet['lpips']),
                'ag_pet': to_python_type(metrics_pet['ag_pred']),
            }
            
            fusion_sample_metrics = {
                'idx': int(idx + 1),
                'filename': filename,
                'mse_fused': to_python_type(fusion_metrics['mse']),
                'mae_fused': to_python_type(fusion_metrics['mae']),
                'rmse_fused': to_python_type(fusion_metrics['rmse']),
                'psnr_fused': to_python_type(fusion_metrics['psnr']),
                'ssim_fused': to_python_type(fusion_metrics['ssim']),
                'vif_fused': to_python_type(fusion_metrics['vif']),
                'lpips_fused': to_python_type(fusion_metrics['lpips']),
                'ag_fused': to_python_type(fusion_metrics['ag_pred']),
            }
            
            all_metrics.append(sample_metrics)
            all_fusion_metrics.append(fusion_sample_metrics)
    
    # ==================== Calculate Statistics ====================
    print("\\n📊 Calculating statistics...")
    
    def calc_stats(metrics_list, key):
        values = [m[key] for m in metrics_list]
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    # Calculate statistics for all metrics
    metric_keys = ['mse', 'mae', 'rmse', 'psnr', 'ssim', 'vif', 'lpips', 'ag']
    modalities = ['t1', 't2', 'pet']
    
    statistics = {}
    for mod in modalities:
        statistics[mod] = {}
        for key in metric_keys:
            full_key = f'{key}_{mod}'
            statistics[mod][key] = calc_stats(all_metrics, full_key)
    
    # Fusion statistics
    statistics['fused'] = {}
    for key in metric_keys:
        full_key = f'{key}_fused'
        statistics['fused'][key] = calc_stats(all_fusion_metrics, full_key)
    
    # ==================== Save Results ====================
    print("💾 Saving results...")
    
    # Save per-sample metrics CSV
    with open(cfg.metrics_csv, 'w', newline='') as f:
        fieldnames = list(all_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    # Save fusion metrics CSV
    with open(cfg.fusion_metrics_csv, 'w', newline='') as f:
        fieldnames = list(all_fusion_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_fusion_metrics)
    
    # Save summary JSON
    summary = {
        'test_info': {
            'model_path': cfg.best_model_path,
            'test_split': cfg.test_split,
            'num_samples': len(all_metrics),
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(cfg.device)
        },
        'statistics': statistics,
        'per_sample_metrics': all_metrics,
        'fusion_metrics': all_fusion_metrics
    }
    
    with open(cfg.summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save summary TXT
    with open(cfg.summary_txt, 'w') as f:
        f.write("="*70 + "\\n")
        f.write(" "*15 + "DIFFUSION SR MODEL - TEST RESULTS\\n")
        f.write("="*70 + "\\n\\n")
        
        f.write(f"Model:        {cfg.best_model_path}\\n")
        f.write(f"Test Split:   {cfg.test_split}\\n")
        f.write(f"Samples:      {len(all_metrics)}\\n")
        f.write(f"Test Date:    {summary['test_info']['test_date']}\\n\\n")
        
        f.write("="*70 + "\\n")
        f.write("OVERALL METRICS (Mean ± Std)\\n")
        f.write("="*70 + "\\n\\n")
        
        for mod in modalities + ['fused']:
            f.write(f"\\n{mod.upper()} Modality:\\n")
            f.write("-"*70 + "\\n")
            for key in metric_keys:
                stats = statistics[mod][key]
                f.write(f"  {key.upper():6s}: {stats['mean']:7.4f} ± {stats['std']:7.4f}  ")
                f.write(f"[{stats['min']:7.4f}, {stats['max']:7.4f}]\\n")
    
    # ==================== Print Summary ====================
    print("\\n" + "="*70)
    print(" "*20 + "FINAL TEST RESULTS")
    print("="*70)
    print(f"Samples: {len(all_metrics)}\\n")
    
    for mod in modalities + ['fused']:
        print(f"\\n{mod.upper()} Metrics (Mean ± Std):")
        print("-"*70)
        for key in ['psnr', 'ssim', 'lpips', 'vif', 'ag']:
            stats = statistics[mod][key]
            print(f"  {key.upper():6s}: {stats['mean']:7.4f} ± {stats['std']:7.4f}")
    
    print("\\n" + "="*70 + "\\n")
    
    print("✅ Results saved:")
    print(f"   📄 Per-sample metrics:  {cfg.metrics_csv}")
    print(f"   📄 Fusion metrics:      {cfg.fusion_metrics_csv}")
    print(f"   📄 Summary JSON:        {cfg.summary_json}")
    print(f"   📄 Summary TXT:         {cfg.summary_txt}")
    print(f"   🖼️  Fused SR images:    {cfg.fused_sr_dir} ({len(all_metrics)} images)")
    print(f"   🖼️  Fused GT images:    {cfg.fused_gt_dir} ({len(all_metrics)} images)")
    print(f"   🖼️  Fused LR images:    {cfg.fused_lr_dir} ({len(all_metrics)} images)")
    
    print("\\n✅ Testing complete!\\n")
    
    return summary

# ============================================================
#                    MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cfg = TestConfig()
    results = test_model(cfg)