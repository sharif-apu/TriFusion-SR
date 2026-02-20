# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bars in Jupyter
from matplotlib import cm # For advanced colormaps
import cv2 # For image processing and blending
import numpy as np # Import numpy for array operations
import math
import random # For random noise in diff
# from tmfa import *

# mport torch
# import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 1. UNCERTAINTY ESTIMATION MODULE
# ============================================================================

class UncertaintyEstimator(nn.Module):
    """
    Estimates aleatoric (data) uncertainty for each modality's features.
    
    Outputs:
        - mean: Expected feature value
        - log_var: Log variance (uncertainty measure)
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Separate heads for mean and variance
        self.mean_head = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        self.log_var_head = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            mean: (B, C, H, W)
            log_var: (B, C, H, W) - log variance for numerical stability
        """
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return mean, log_var


# ============================================================================
# 2. UNCERTAINTY-GUIDED FUSION BLOCK
# ============================================================================

class UncertaintyGuidedFusion(nn.Module):
    """
    Fuses multi-modal features using uncertainty-based weighting.
    
    Key Idea: 
        - High uncertainty → Low weight (don't trust this modality)
        - Low uncertainty → High weight (trust this modality)
    """
    def __init__(self, in_channels, num_modalities=3, fusion_type='learned'):
        """
        Args:
            in_channels: Channels per modality
            num_modalities: Number of input modalities (default: 3 for T1/T2/PET)
            fusion_type: 'learned' or 'inverse_variance'
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        # Uncertainty estimators for each modality
        self.uncertainty_estimators = nn.ModuleList([
            UncertaintyEstimator(in_channels) for _ in range(num_modalities)
        ])
        
        if fusion_type == 'learned':
            # Learnable fusion weights conditioned on uncertainty
            self.weight_generator = nn.Sequential(
                nn.Conv2d(in_channels * num_modalities * 2, in_channels * num_modalities, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * num_modalities, num_modalities, 1),
                nn.Softmax(dim=1)  # Normalize across modalities
            )
        
        # Optional: Confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of [feat_t1, feat_t2, feat_pet], each (B, C, H, W)
        Returns:
            fused_features: (B, C, H, W)
            uncertainty_map: (B, num_modalities, H, W) - for visualization
        """
        assert len(modality_features) == self.num_modalities
        
        means = []
        log_vars = []
        
        # Estimate uncertainty for each modality
        for i, feat in enumerate(modality_features):
            mean, log_var = self.uncertainty_estimators[i](feat)
            means.append(mean)
            log_vars.append(log_var)
        
        # Stack for batch processing
        means = torch.stack(means, dim=1)  # (B, M, C, H, W)
        log_vars = torch.stack(log_vars, dim=1)  # (B, M, C, H, W)
        
        if self.fusion_type == 'inverse_variance':
            # Classic inverse variance weighting
            variances = torch.exp(log_vars)  # (B, M, C, H, W)
            precisions = 1.0 / (variances + 1e-8)  # Inverse variance
            
            # Normalize weights across modalities
            weights = precisions / (precisions.sum(dim=1, keepdim=True) + 1e-8)
            
            # Weighted fusion
            fused = (means * weights).sum(dim=1)  # (B, C, H, W)
            
            # Uncertainty map (average variance across channels)
            uncertainty_map = variances.mean(dim=2)  # (B, M, H, W)
            
        elif self.fusion_type == 'learned':
            # Learned fusion conditioned on means and uncertainties
            B, M, C, H, W = means.shape
            
            # Concatenate means and log_vars as conditioning
            conditioning = torch.cat([
                means.view(B, M * C, H, W),
                log_vars.view(B, M * C, H, W)
            ], dim=1)  # (B, 2*M*C, H, W)
            
            # Generate fusion weights
            weights = self.weight_generator(conditioning)  # (B, M, 1, H, W)
            weights = weights.unsqueeze(2)  # (B, M, 1, H, W)
            
            # Apply temperature scaling for calibration
            weights = weights / self.temperature
            weights = F.softmax(weights, dim=1)
            
            # Weighted fusion
            fused = (means * weights).sum(dim=1)  # (B, C, H, W)
            
            # Uncertainty map
            uncertainty_map = torch.exp(log_vars).mean(dim=2)  # (B, M, H, W)
        
        return fused, uncertainty_map


# ============================================================================
# 3. ENHANCED ASFE WITH UNCERTAINTY
# ============================================================================

class ASFE_Block_with_Uncertainty(nn.Module):
    """
    Enhanced ASFE that incorporates uncertainty information.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, 
                                      padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

        # Uncertainty-aware gating (NEW!)
        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),  # +1 for uncertainty map
            nn.Sigmoid()
        )
        
        # Adaptive Gating
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, uncertainty_map=None):
        """
        Args:
            x: Input features (B, C, H, W)
            uncertainty_map: Optional uncertainty map (B, 1, H, W)
        """
        identity = x
        
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attention_map = torch.sigmoid(avg_out + max_out)
        x_ca = x * channel_attention_map.expand_as(x)
        
        # Spatial Attention
        avg_out_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_attention_map = self.sigmoid_spatial(self.conv_spatial(spatial_features))
        x_sa = x_ca * spatial_attention_map
        
        # Uncertainty-aware modulation (NEW!)
        if uncertainty_map is not None:
            # Resize uncertainty map if needed
            if uncertainty_map.shape[2:] != x_sa.shape[2:]:
                uncertainty_map = F.interpolate(uncertainty_map, size=x_sa.shape[2:], 
                                               mode='bilinear', align_corners=False)
            
            # Concatenate and generate uncertainty gate
            uncertainty_input = torch.cat([x_sa, uncertainty_map], dim=1)
            uncertainty_gate = self.uncertainty_gate(uncertainty_input)
            x_sa = x_sa * uncertainty_gate

        # Adaptive Gating
        combined_features_for_gate = torch.cat([identity, x_sa], dim=1)
        gate_weights = self.gate_conv(combined_features_for_gate)
        
        w1 = gate_weights[:, 0:1, :, :]
        w2 = gate_weights[:, 1:2, :, :]
        
        output = w1 * identity + w2 * x_sa
        output = output + self.gamma * identity
        
        return output


# ============================================================================
# 4. UNCERTAINTY LOSS
# ============================================================================

class UncertaintyLoss(nn.Module):
    """
    Loss function that penalizes both prediction error and uncertainty.
    
    L = (1/2) * exp(-log_var) * ||pred - target||^2 + (1/2) * log_var
    
    This encourages:
        - Low uncertainty when predictions are accurate
        - High uncertainty when predictions are inaccurate
    """
    def __init__(self, lambda_uncertainty=0.1):
        super().__init__()
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(self, pred_mean, pred_log_var, target):
        """
        Args:
            pred_mean: Predicted mean (B, C, H, W)
            pred_log_var: Predicted log variance (B, C, H, W)
            target: Ground truth (B, C, H, W)
        """
        # Reconstruction loss weighted by uncertainty
        precision = torch.exp(-pred_log_var)
        recon_loss = 0.5 * precision * F.mse_loss(pred_mean, target, reduction='none')
        
        # Regularization term (prevents collapsing to high uncertainty)
        reg_loss = 0.5 * pred_log_var
        
        # Total loss
        total_loss = (recon_loss + self.lambda_uncertainty * reg_loss).mean()
        
        return total_loss

# integration_code = '''
# ============================================================================
# UPDATED: Wavelet_ASFE_Fusion_Model with Uncertainty
# ============================================================================

class Wavelet_ASFE_Fusion_Model_with_Uncertainty(nn.Module):
    """
    Enhanced version with uncertainty-guided fusion.
    
    Key Changes:
    1. Added UncertaintyGuidedFusion after wavelet decomposition
    2. Replaced ASFE_Block with ASFE_Block_with_Uncertainty
    3. Returns uncertainty maps for visualization/loss computation
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_c = cfg.encoder_channels

        # Modality Encoders (unchanged)
        self.t1_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.pet_encoder = nn.Sequential(
            nn.Conv2d(3, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        
        # Wavelet Decomposition Blocks (unchanged)
        self.t1_wdb = Wavelet_Decomposition_Block(enc_c)
        self.t2_wdb = Wavelet_Decomposition_Block(enc_c)
        self.pet_wdb = Wavelet_Decomposition_Block(enc_c)

        # 🔥 NEW: Uncertainty-Guided Fusion for LF and HF separately
        self.lf_uncertainty_fusion = UncertaintyGuidedFusion(
            in_channels=enc_c, 
            num_modalities=3, 
            fusion_type='learned'  # or 'inverse_variance'
        )
        self.hf_uncertainty_fusion = UncertaintyGuidedFusion(
            in_channels=enc_c, 
            num_modalities=3, 
            fusion_type='learned'
        )

        # 🔥 UPDATED: Use uncertainty-aware ASFE
        self.asfe_input_channels = enc_c * 2  # Now only 2 (LF + HF fused)
        self.asfe_block = ASFE_Block_with_Uncertainty(
            in_channels=self.asfe_input_channels, 
            reduction_ratio=16
        )

        # CMF Reduction (updated input channels)
        self.cmf_reduce_conv = nn.Sequential(
            nn.Conv2d(self.asfe_input_channels, enc_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(enc_c),
            nn.ReLU(inplace=True)
        )

        # Recomposition (unchanged)
        self.lf_reduction = nn.Conv2d(enc_c, enc_c, kernel_size=3, stride=2, padding=1)
        self.final_recomp = Wavelet_Recomposition_Block(in_channels=enc_c)
        
        # Final mapping
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_c, cfg.conditional_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_lr, t2_lr, pet_lr, return_uncertainty=False):
        """
        Args:
            t1_lr, t2_lr, pet_lr: Input modalities
            return_uncertainty: If True, returns uncertainty maps for visualization
        
        Returns:
            conditional_features: (B, C, H, W)
            uncertainty_dict: (optional) Dictionary with uncertainty maps
        """
        # 1. Encode Features
        t1_f = self.t1_encoder(t1_lr)
        t2_f = self.t2_encoder(t2_lr)
        pet_f = self.pet_encoder(pet_lr)

        # 2. Wavelet Decomposition
        t1_lf, t1_hf = self.t1_wdb(t1_f)
        t2_lf, t2_hf = self.t2_wdb(t2_f)
        pet_lf, pet_hf = self.pet_wdb(pet_f)
        
        # 3. 🔥 Uncertainty-Guided Fusion (separate for LF and HF)
        lf_fused, lf_uncertainty = self.lf_uncertainty_fusion([t1_lf, t2_lf, pet_lf])
        hf_fused, hf_uncertainty = self.hf_uncertainty_fusion([t1_hf, t2_hf, pet_hf])
        
        # 4. Upscale LF to match HF resolution
        _, _, h, w = hf_fused.shape
        lf_fused_up = F.interpolate(lf_fused, size=(h, w), mode='bilinear', align_corners=False)
        
        # 5. Concatenate fused LF and HF
        full_features = torch.cat([lf_fused_up, hf_fused], dim=1)  # (B, 2*C, H, W)
        
        # 6. Compute combined uncertainty map for ASFE
        # Average uncertainty across modalities and combine LF/HF uncertainties
        lf_uncertainty_avg = lf_uncertainty.mean(dim=1, keepdim=True)  # (B, 1, H/2, W/2)
        hf_uncertainty_avg = hf_uncertainty.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Upscale LF uncertainty to match HF
        lf_uncertainty_up = F.interpolate(lf_uncertainty_avg, size=(h, w), 
                                         mode='bilinear', align_corners=False)
        
        # Combine uncertainties (average or max)
        combined_uncertainty = (lf_uncertainty_up + hf_uncertainty_avg) / 2.0
        
        # 7. 🔥 ASFE with Uncertainty Modulation
        enhanced_features = self.asfe_block(full_features, combined_uncertainty)
        
        # 8. CMF Reduction
        cmf_output = self.cmf_reduce_conv(enhanced_features)
        
        # 9. Recomposition
        lf_component = self.lf_reduction(cmf_output)
        hf_component = cmf_output
        recomposed_features = self.final_recomp(lf_component, hf_component)
        
        # 10. Final mapping
        conditional_features = self.final_conv(recomposed_features)

        if return_uncertainty:
            uncertainty_dict = {
                'lf_uncertainty': lf_uncertainty,  # (B, 3, H/2, W/2)
                'hf_uncertainty': hf_uncertainty,  # (B, 3, H, W)
                'combined_uncertainty': combined_uncertainty  # (B, 1, H, W)
            }
            return conditional_features, uncertainty_dict
        
        return conditional_features


# ============================================================================
# ALTERNATIVE: Simpler Integration (Uncertainty only at final fusion)
# ============================================================================

class Wavelet__Rectified_ASFE_Fusion_Model(nn.Module):
    """
    Simpler version: Apply uncertainty fusion only after ASFE.
    Good for initial experiments.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_c = cfg.encoder_channels

        # Keep original architecture mostly unchanged
        self.t1_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.pet_encoder = nn.Sequential(
            nn.Conv2d(3, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        
        self.t1_wdb = Wavelet_Decomposition_Block(enc_c)
        self.t2_wdb = Wavelet_Decomposition_Block(enc_c)
        self.pet_wdb = Wavelet_Decomposition_Block(enc_c)

        self.asfe_input_channels = enc_c * 6
        self.asfe_block = ASFE_Block(in_channels=self.asfe_input_channels, reduction_ratio=16)

        # 🔥 NEW: Add uncertainty estimation AFTER ASFE
        self.uncertainty_estimator = UncertaintyEstimator(self.asfe_input_channels)

        self.cmf_reduce_conv = nn.Sequential(
            nn.Conv2d(self.asfe_input_channels, enc_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(enc_c),
            nn.ReLU(inplace=True)
        )

        self.lf_reduction = nn.Conv2d(enc_c, enc_c, kernel_size=3, stride=2, padding=1)
        self.final_recomp = Wavelet_Recomposition_Block(in_channels=enc_c)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_c, cfg.conditional_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_lr, t2_lr, pet_lr, return_uncertainty=False):
        # Original forward pass
        t1_f = self.t1_encoder(t1_lr)
        t2_f = self.t2_encoder(t2_lr)
        pet_f = self.pet_encoder(pet_lr)

        t1_lf, t1_hf = self.t1_wdb(t1_f)
        t2_lf, t2_hf = self.t2_wdb(t2_f)
        pet_lf, pet_hf = self.pet_wdb(pet_f)
        
        _, _, h, w = t1_hf.shape
        t1_lf_up = F.interpolate(t1_lf, size=(h, w), mode='nearest')
        t2_lf_up = F.interpolate(t2_lf, size=(h, w), mode='nearest')
        pet_lf_up = F.interpolate(pet_lf, size=(h, w), mode='nearest')
        
        full_features = torch.cat([t1_lf_up, t1_hf, t2_lf_up, t2_hf, pet_lf_up, pet_hf], dim=1)

        # 🔥 NEW: Estimate uncertainty
        enhanced_features_mean, enhanced_features_log_var = self.uncertainty_estimator(full_features)
        
        # Use mean for downstream processing
        enhanced_features = self.asfe_block(enhanced_features_mean)
        
        cmf_output = self.cmf_reduce_conv(enhanced_features)
        lf_component = self.lf_reduction(cmf_output)
        hf_component = cmf_output
        recomposed_features = self.final_recomp(lf_component, hf_component)
        conditional_features = self.final_conv(recomposed_features)

        if return_uncertainty:
            uncertainty_map = torch.exp(enhanced_features_log_var).mean(dim=1, keepdim=True)
            return conditional_features, uncertainty_map
        
        return conditional_features
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_Simple_Fusion(nn.Module):
    """
    Ablation Baseline: 'Vanilla' Fusion.
    - No Wavelet Transform (Spatial domain only)
    - No Uncertainty Estimation (Deterministic)
    - No ASFE (Simple Concatenation)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_c = cfg.encoder_channels

        # 1. Encoders (Keep same as original for fair comparison)
        self.t1_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        self.pet_encoder = nn.Sequential(
            nn.Conv2d(3, enc_c, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True)
        )
        
        # 2. Simple Fusion Block (Replaces ASFE & Wavelet mixing)
        # We concatenate 3 modalities (T1, T2, PET), so input is enc_c * 3
        self.fusion_input_channels = enc_c * 3
        
        self.simple_fusion_conv = nn.Sequential(
            nn.Conv2d(self.fusion_input_channels, enc_c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(enc_c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_c * 2, enc_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(enc_c),
            nn.ReLU(inplace=True)
        )

        # 3. Final Projection (Matches output dimension of original)
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_c, cfg.conditional_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_lr, t2_lr, pet_lr, return_uncertainty=False):
        # 1. Encode Features
        t1_f = self.t1_encoder(t1_lr)
        t2_f = self.t2_encoder(t2_lr)
        pet_f = self.pet_encoder(pet_lr)

        # 2. Naive Fusion (Concatenation)
        # Shape: [B, C*3, H, W]
        combined_features = torch.cat([t1_f, t2_f, pet_f], dim=1)

        # 3. Process Fused Features
        # Replaces ASFE and Wavelet processing
        fused_map = self.simple_fusion_conv(combined_features)
        
        # 4. Final Output
        conditional_features = self.final_conv(fused_map)

        # Handle interface compatibility (return None for uncertainty)
        if return_uncertainty:
            return conditional_features, None
        
        return conditional_features

# '''

# print(integration_code)
# print("\n" + "="*80)
# print("✅ Two integration options provided:")
# print("1. Full Integration - Uncertainty at LF/HF fusion + ASFE")
# print("2. Simple Integration - Uncertainty only after ASFE (easier to start)")

class Wavelet_Decomposition_Block(nn.Module):
    """
    Simulates one level of DWT by separating features into 
    Low-Frequency (LF) via stride-2 Conv and High-Frequency (HF) via stride-1 Conv.
    """
    def __init__(self, in_channels):
        super().__init__()
        # LF component: Downscaled (H/2, W/2) - Captures approximation/low-frequency features
        self.lf_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        # HF component: Full-scale (H, W) - Captures detail/high-frequency features
        self.hf_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        lf_features = self.lf_conv(x)  # (B, C, H/2, W/2)
        hf_features = self.hf_conv(x)  # (B, C, H, W)
        return lf_features, hf_features


# --- 3. Wavelet Recomposition Block (Approximation) ---
class Wavelet_Recomposition_Block(nn.Module):
    """
    Simulates one level of IDWT by upscaling LF features and blending with HF features.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Upscale LF component using Transposed Convolution
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        # Final conv for blending upscaled LF and the full-res HF features
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lf_features, hf_features):
        lf_upscaled = self.upsample(lf_features) # Upscale to (B, C, H, W)
        
        # Crop the upscaled LF features to match the HF features' resolution (H, W)
        _, _, h, w = hf_features.shape
        lf_upscaled = lf_upscaled[:, :, :h, :w] 

        combined = torch.cat([lf_upscaled, hf_features], dim=1)
        recomposed = self.relu(self.final_conv(combined))
        return recomposed


# --- 4. ASFE-Block (Adaptive Spatial-Feature Enhancement) ---
class ASFE_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid() 

        # Adaptive Gating
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False), 
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable residual weight

    def forward(self, x):
        identity = x
        
        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x)); max_out = self.mlp(self.max_pool(x))
        channel_attention_map = torch.sigmoid(avg_out + max_out) 
        x_ca = x * channel_attention_map.expand_as(x)
        
        # Spatial Attention
        avg_out_spatial = torch.mean(x_ca, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_attention_map = self.sigmoid_spatial(self.conv_spatial(spatial_features))
        x_sa = x_ca * spatial_attention_map

        # Adaptive Gating
        combined_features_for_gate = torch.cat([identity, x_sa], dim=1)
        gate_weights = self.gate_conv(combined_features_for_gate)
        
        w1 = gate_weights[:, 0:1, :, :]; w2 = gate_weights[:, 1:2, :, :]
        
        output = w1 * identity + w2 * x_sa
        output = output + self.gamma * identity 
        return output
    

# --- 5. Wavelet-ASFE Fusion Model (Main Architecture) ---
class Wavelet_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_c = cfg.encoder_channels

        # Modality Encoders (Initial feature extraction)
        self.t1_encoder = nn.Sequential(nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        self.t2_encoder = nn.Sequential(nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        self.pet_encoder = nn.Sequential(nn.Conv2d(3, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        
        # Wavelet Decomposition Blocks
        self.t1_wdb = Wavelet_Decomposition_Block(enc_c)
        self.t2_wdb = Wavelet_Decomposition_Block(enc_c)
        self.pet_wdb = Wavelet_Decomposition_Block(enc_c)

        # ASFE Input Channels: 3 modalities * 2 scales * enc_c = 6 * enc_c
        self.asfe_input_channels = enc_c * 6 
        
        # ASFE Block: Fuses and enhances all 6 feature maps at once
        #self.asfe_block = ASFE_Block(in_channels=self.asfe_input_channels, reduction_ratio=16)

        # CMF Reduction: Reduce channels from 6*enc_c back down to enc_c (Post-ASFE)
        self.cmf_reduce_conv = nn.Sequential(
            nn.Conv2d(self.asfe_input_channels, enc_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(enc_c),
            nn.ReLU(inplace=True)
        )

        # Recomposition Preparation: Downsample CMF output to serve as the LF input for IDWT
        self.lf_reduction = nn.Conv2d(enc_c, enc_c, kernel_size=3, stride=2, padding=1) 
        self.final_recomp = Wavelet_Recomposition_Block(in_channels=enc_c)
        
        # Final mapping to conditional_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_c, cfg.conditional_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_lr, t2_lr, pet_lr):
        # 1. Encode Features
        t1_f = self.t1_encoder(t1_lr)
        t2_f = self.t2_encoder(t2_lr)
        pet_f = self.pet_encoder(pet_lr)

        # 2. Wavelet Decomposition
        t1_lf, t1_hf = self.t1_wdb(t1_f)
        t2_lf, t2_hf = self.t2_wdb(t2_f)
        pet_lf, pet_hf = self.pet_wdb(pet_f)
        
        # 3. Prepare for ASFE Fusion: Upscale all LF features to match HF size for unified fusion
        _, _, h, w = t1_hf.shape
        t1_lf_up = F.interpolate(t1_lf, size=(h, w), mode='nearest')
        t2_lf_up = F.interpolate(t2_lf, size=(h, w), mode='nearest')
        pet_lf_up = F.interpolate(pet_lf, size=(h, w), mode='nearest')
        
        # Concatenate all 6 feature maps (3 modalities * 2 scales)
        full_features = torch.cat([t1_lf_up, t1_hf, t2_lf_up, t2_hf, pet_lf_up, pet_hf], dim=1)
        # Shape: (B, 6*C, H, W)

        # 4. ASFE Fusion and Enhancement
        #enhanced_features = self.asfe_block(full_features)
        
        # 5. CMF Reduction
        cmf_output = self.cmf_reduce_conv(full_features) # (B, C, H, W)
        
        # 6. Recomposition Preparation (Separate CMF output back into LF/HF components)
        lf_component = self.lf_reduction(cmf_output) # LF: (B, C, H/2, W/2)
        hf_component = cmf_output                     # HF: (B, C, H, W)

        # 7. Final Wavelet Recomposition
        recomposed_features = self.final_recomp(lf_component, hf_component)
        
        # 8. Final mapping
        conditional_features = self.final_conv(recomposed_features)

        return conditional_features

# --- 5. Wavelet-ASFE Fusion Model (Main Architecture) ---
class Wavelet_ASFE_Fusion_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_c = cfg.encoder_channels

        # Modality Encoders (Initial feature extraction)
        self.t1_encoder = nn.Sequential(nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        self.t2_encoder = nn.Sequential(nn.Conv2d(1, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        self.pet_encoder = nn.Sequential(nn.Conv2d(3, enc_c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(enc_c, enc_c, 3, 1, 1), nn.ReLU(inplace=True))
        
        # Wavelet Decomposition Blocks
        self.t1_wdb = Wavelet_Decomposition_Block(enc_c)
        self.t2_wdb = Wavelet_Decomposition_Block(enc_c)
        self.pet_wdb = Wavelet_Decomposition_Block(enc_c)

        # ASFE Input Channels: 3 modalities * 2 scales * enc_c = 6 * enc_c
        self.asfe_input_channels = enc_c * 6 
        
        # ASFE Block: Fuses and enhances all 6 feature maps at once
        self.asfe_block = ASFE_Block(in_channels=self.asfe_input_channels, reduction_ratio=16)

        # CMF Reduction: Reduce channels from 6*enc_c back down to enc_c (Post-ASFE)
        self.cmf_reduce_conv = nn.Sequential(
            nn.Conv2d(self.asfe_input_channels, enc_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(enc_c),
            nn.ReLU(inplace=True)
        )

        # Recomposition Preparation: Downsample CMF output to serve as the LF input for IDWT
        self.lf_reduction = nn.Conv2d(enc_c, enc_c, kernel_size=3, stride=2, padding=1) 
        self.final_recomp = Wavelet_Recomposition_Block(in_channels=enc_c)
        
        # Final mapping to conditional_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_c, cfg.conditional_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1_lr, t2_lr, pet_lr):
        # 1. Encode Features
        t1_f = self.t1_encoder(t1_lr)
        t2_f = self.t2_encoder(t2_lr)
        pet_f = self.pet_encoder(pet_lr)

        # 2. Wavelet Decomposition
        t1_lf, t1_hf = self.t1_wdb(t1_f)
        t2_lf, t2_hf = self.t2_wdb(t2_f)
        pet_lf, pet_hf = self.pet_wdb(pet_f)
        
        # 3. Prepare for ASFE Fusion: Upscale all LF features to match HF size for unified fusion
        _, _, h, w = t1_hf.shape
        t1_lf_up = F.interpolate(t1_lf, size=(h, w), mode='nearest')
        t2_lf_up = F.interpolate(t2_lf, size=(h, w), mode='nearest')
        pet_lf_up = F.interpolate(pet_lf, size=(h, w), mode='nearest')
        
        # Concatenate all 6 feature maps (3 modalities * 2 scales)
        full_features = torch.cat([t1_lf_up, t1_hf, t2_lf_up, t2_hf, pet_lf_up, pet_hf], dim=1)
        # Shape: (B, 6*C, H, W)

        # 4. ASFE Fusion and Enhancement
        enhanced_features = self.asfe_block(full_features)
        
        # 5. CMF Reduction
        cmf_output = self.cmf_reduce_conv(enhanced_features) # (B, C, H, W)
        
        # 6. Recomposition Preparation (Separate CMF output back into LF/HF components)
        lf_component = self.lf_reduction(cmf_output) # LF: (B, C, H/2, W/2)
        hf_component = cmf_output                     # HF: (B, C, H, W)

        # 7. Final Wavelet Recomposition
        recomposed_features = self.final_recomp(lf_component, hf_component)
        
        # 8. Final mapping
        conditional_features = self.final_conv(recomposed_features)

        return conditional_features
        

# %%
# --- Sinusoidal Positional Embeddings for Timesteps ---
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# --- GroupNorm wrapper (unchanged) ---
class GroupNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        num_groups = min(8, channels)
        if num_groups == 0: num_groups = 1
        if channels % num_groups != 0:
            for i in range(num_groups, 0, -1):
                if channels % i == 0:
                    num_groups = i
                    break
            if num_groups == 0: num_groups = 1
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-5)

    def forward(self, x):
        return self.gn(x)

# --- ResBlock (fixed conditional feature handling) ---
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, conditional_channels=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv1 = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.conv2 = nn.Sequential(
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.cond_proj = nn.Conv2d(conditional_channels, out_channels, 1) if conditional_channels is not None else None

    def forward(self, x, time_emb, conditional_features=None):
        h = self.conv1(x)
        h = h + self.mlp(time_emb)[:, :, None, None]

        if self.cond_proj is not None and conditional_features is not None:
            # --- Defensive fix for conditional_features ---
            if conditional_features.dim() == 5:
                conditional_features = conditional_features.squeeze(2)
            if h.shape[2] != conditional_features.shape[2] or h.shape[3] != conditional_features.shape[3]:
                conditional_features = F.interpolate(
                    conditional_features, size=(h.shape[2], h.shape[3]),
                    mode='bilinear', align_corners=False
                )
            proj_cond = self.cond_proj(conditional_features)
            if proj_cond.shape != h.shape:
                raise RuntimeError(f"Projected conditional features shape {proj_cond.shape} does not match h shape {h.shape}")
            h = h + proj_cond

        h = self.conv2(h)
        return h + self.residual_conv(x)

# --- AttentionBlock (unchanged) ---
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = GroupNorm(channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

        self.num_heads = 8
        if channels < self.num_heads:
            self.num_heads = channels
        while channels % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        if self.num_heads == 0: self.num_heads = 1

        self.head_dim = max(1, channels // self.num_heads)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        h = self.group_norm(x)
        h = self.proj_in(h)

        b, c, H, W = h.shape
        # recompute usable heads in case c changed
        current_num_heads = self.num_heads
        while c % current_num_heads != 0 and current_num_heads > 1:
            current_num_heads -= 1
        if current_num_heads == 0: current_num_heads = 1
        current_head_dim = c // current_num_heads

        h = h.view(b, current_num_heads, current_head_dim, H * W)
        q, k, v = h, h, h

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        attn_output = attn_output.view(b, c, H, W)
        h = self.proj_out(attn_output)
        return x + h

# --- DownBlock (unchanged, returns skip and downsampled output). ---
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attn, conditional_channels=None, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim, conditional_channels)
            for i in range(num_res_blocks)
        ])
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, conditional_features=None):
        for res_block in self.res_blocks:
            x = res_block(x, time_emb, conditional_features)
        x_skipped = self.attn(x)
        return x_skipped, self.downsample(x_skipped)

# --- UpBlock (FIXED) ---
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, has_attn, conditional_channels=None, num_res_blocks=2):
        """
        in_channels: channels of the tensor coming into this up block (from previous stage)
        skip_channels: channels of the skip connection from the corresponding down block
        out_channels: desired output channels from this block
        """
        super().__init__()
        # Upsample from in_channels -> out_channels (spatially doubled, channels reduced to out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        # After upsample, we will concatenate with skip (out_channels + skip_channels),
        # so the first ResBlock input channels = out_channels + skip_channels -> maps to out_channels
        self.res_blocks = nn.ModuleList([
            ResBlock(out_channels + skip_channels if i == 0 else out_channels, out_channels, time_emb_dim, conditional_channels)
            for i in range(num_res_blocks)
        ])
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip_connection, time_emb, conditional_features=None):
        x = self.upsample(x)  # now has out_channels channels
        # Ensure skip has the same spatial size
        if x.shape[2:] != skip_connection.shape[2:]:
            skip_connection = F.interpolate(skip_connection, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)  # [B, out_channels + skip_channels, H, W]
        for res_block in self.res_blocks:
            x = res_block(x, time_emb, conditional_features)
        x = self.attn(x)
        return x

# --- U-Net (updated construction to pass skip_channels correctly) ---
class U_Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        image_channels = cfg.image_channels_hr
        time_emb_dim = cfg.base_channels * 4

        # Keep current time_mlp design but ensure dimensions line up:
        # Sinusoidal returns length "time_emb_dim // 2" in your original pattern,
        # then linear expands to time_emb_dim.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.initial_conv = nn.Conv2d(image_channels, cfg.base_channels, 3, padding=1)

        # Build downs and remember out_channels for each down (for skip_channels)
        self.downs = nn.ModuleList([])
        down_out_channels = []  # store skip channel sizes
        in_channels = cfg.base_channels
        current_res = cfg.image_size_hr
        for multiplier in cfg.channel_multipliers:
            out_channels = cfg.base_channels * multiplier
            has_attn = current_res in cfg.attn_resolutions
            # Use cfg.num_res_blocks as before
            self.downs.append(DownBlock(in_channels, out_channels, time_emb_dim, has_attn, cfg.conditional_channels, num_res_blocks=cfg.num_res_blocks))
            down_out_channels.append(out_channels)
            in_channels = out_channels
            current_res //= 2

        # Middle blocks
        self.middle_block1 = ResBlock(in_channels, in_channels, time_emb_dim, cfg.conditional_channels)
        self.middle_attn = AttentionBlock(in_channels)
        self.middle_block2 = ResBlock(in_channels, in_channels, time_emb_dim, cfg.conditional_channels)

        # Build ups using stored skip channels. We'll track prev_channels = channels coming from middle.
        self.ups = nn.ModuleList([])
        prev_channels = in_channels  # channels from middle (deepest)
        # iterate backwards over channel_multipliers and down_out_channels
        for i in reversed(range(len(cfg.channel_multipliers))):
            out_channels = cfg.base_channels * cfg.channel_multipliers[i]
            skip_channels = down_out_channels[i]
            has_attn = (out_channels * 2) in cfg.attn_resolutions if hasattr(cfg, 'attn_resolutions') else False
            # Create UpBlock(prev_channels, skip_channels, out_channels)
            self.ups.append(UpBlock(prev_channels, skip_channels, out_channels, time_emb_dim, has_attn, cfg.conditional_channels, num_res_blocks=cfg.num_res_blocks))
            prev_channels = out_channels  # next up will produce this many channels

        self.final_conv = nn.Sequential(
            GroupNorm(cfg.base_channels),
            nn.SiLU(),
            nn.Conv2d(cfg.base_channels, image_channels, 3, padding=1)
        )

    def forward(self, x, time, conditional_features):
        time_emb = self.time_mlp(time)  # shape [B, time_emb_dim]
        x = self.initial_conv(x)

        # Collect skip connections (the skip is the pre-downsample feature)
        skip_connections = []
        for down_block in self.downs:
            skip, x = down_block(x, time_emb, conditional_features)
            skip_connections.append(skip)

        # Middle
        x = self.middle_block1(x, time_emb, conditional_features)
        x = self.middle_attn(x)
        x = self.middle_block2(x, time_emb, conditional_features)

        # Ups: use skips in reverse order
        for up_block in self.ups:
            skip = skip_connections.pop()  # last added skip corresponds to current up
            x = up_block(x, skip, time_emb, conditional_features)

        return self.final_conv(x)

# %%
# --- DDPM Scheduler ---
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM_Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.timesteps = cfg.timesteps
        self.device = cfg.device

        # Define beta schedule
        if cfg.beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps=cfg.timesteps, start=cfg.start_beta, end=cfg.end_beta).to(self.device)
        elif cfg.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps=cfg.timesteps).to(self.device)
        else:
            raise ValueError(f"Unknown beta schedule: {cfg.beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # For calculations involving previous step

        # Register buffers (using setattr to mimic nn.Module's register_buffer for convenience)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_t-1 | x_t, x_0)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
    def get_index_from_list(self, vals, t, x_shape):
        """
        Safely gathers values (e.g., betas, alphas) for a batch of timesteps.
        Returns shape: (batch_size, 1, 1, 1) to broadcast to img shape.
        """
        # Ensure both tensors are on the same device
        vals = vals.to(t.device)

        # Gather corresponding values for each timestep in the batch
        out = vals.gather(0, t)  # t shape: (B,)

        # Reshape for broadcasting (e.g. [B, 1, 1, 1] for 4D images)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))



    def q_sample(self, x_start, t, noise=None):
        """
        Noisy image at time t given x_0 (x_start)
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1-alpha_cumprod_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, conditional_features):
        """
        Sample from p(x_t-1 | x_t) using the predicted noise from the U-Net model.
        """
        beta_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(1. / self.alphas.sqrt(), t, x.shape)

        # Predict noise
        predicted_noise = model(x, t, conditional_features)

        # Use noise to predict x_0
        model_mean = sqrt_recip_alphas_t * (x - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0: # If at the first timestep (t=0), return the predicted x_0
            return model_mean
        else:
            posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, conditional_features):
        """
        Generates a sample image by iteratively denoisng from pure noise.
        """
        img = torch.randn(shape, device=self.device) * 0.5 # Start with pure noise, scaled
        
        # The conditional_features are already LR, and the U-Net's ResBlocks handle their upsampling internally.
        # So no explicit upsampling of conditional_features is needed here before passing to the U-Net.

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t, conditional_features)
        
        # Ensure output is clamped to [-1, 1] range after sampling
        img = torch.clamp(img, -1., 1.)
        return img

    def p_losses(self, model, x_start, t, conditional_features, noise=None):
        """
        Calculates the diffusion loss given x_0, timestep t, and conditional features.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t, conditional_features)

        if self.cfg.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.cfg.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif self.cfg.loss_type == 'huber':
            loss = F.huber_loss(noise, predicted_noise)
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.loss_type}")

        return loss



# %%
# --- Diffusion Super-Resolution Model ---



if __name__=="__main__":

    # --- Initialize and test DiffusionSRModel (fixed) ---
    print("\nInitializing DiffusionSRModel...")
    diffusion_sr_model = DiffusionSRModel(cfg).to(cfg.device)
    print(f"DiffusionSRModel Total Parameters: {sum(p.numel() for p in diffusion_sr_model.parameters()):,}")
