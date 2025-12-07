import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """Комбинированная функция потерь для downscaling"""
    
    def __init__(self, mse_weight=1.0, mae_weight=0.5, spectral_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.spectral_weight = spectral_weight
        
    def forward(self, y_pred, y_true):
        mse_loss = F.mse_loss(y_pred, y_true)
        mae_loss = F.l1_loss(y_pred, y_true)
        
        # Spectral loss (сохранение частотных характеристик)
        spectral_loss = 0.0
        if self.spectral_weight > 0:
            pred_fft = torch.fft.fft2(y_pred)
            true_fft = torch.fft.fft2(y_true)
            spectral_loss = F.mse_loss(pred_fft.real, true_fft.real) + \
                           F.mse_loss(pred_fft.imag, true_fft.imag)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.mae_weight * mae_loss + 
                     self.spectral_weight * spectral_loss)
        
        return total_loss

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss для Neural ODE"""
    
    def __init__(self, data_weight=1.0, physics_weight=0.1, continuity_weight=0.01):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.continuity_weight = continuity_weight
        
    def forward(self, y_pred, y_true, derivatives_pred=None, derivatives_true=None):
        # Data loss
        data_loss = F.mse_loss(y_pred, y_true)
        
        # Physics loss (если есть производные)
        physics_loss = 0.0
        if derivatives_pred is not None and derivatives_true is not None:
            physics_loss = F.mse_loss(derivatives_pred, derivatives_true)
        
        # Continuity loss (для сохранения физической согласованности)
        continuity_loss = 0.0
        if y_pred.shape[1] >= 2:  # как минимум 2 переменные
            # Простая проверка на сохранение энергии/массы
            energy_pred = torch.mean(y_pred ** 2, dim=1)
            energy_true = torch.mean(y_true ** 2, dim=1)
            continuity_loss = F.mse_loss(energy_pred, energy_true)
        
        total_loss = (self.data_weight * data_loss +
                     self.physics_weight * physics_loss +
                     self.continuity_weight * continuity_loss)
        
        return total_loss