import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, with_frelu=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = None
        if with_frelu:
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.double_conv(x)
        if self.relu:
            x = self.relu(x)
        return x

class DownWithResidual(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=1, scale=2):
        super().__init__()
        self.n_layers = n_layers
        self.pool = nn.AvgPool2d(scale)
        self.conv_layers = nn.ModuleList(
            [DoubleConv(in_channels, in_channels) for _ in range(n_layers)]
        )
        self.last_conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.pool(x)
        for i in range(self.n_layers):
            x = self.conv_layers[i](x) + x
        return self.last_conv(x)

class UpWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class WeatherUNet(nn.Module):
    def __init__(self, in_chans=5, out_chans=5, hidden_dim=32, scale_factor=4):
        super().__init__()
        self.target_height = 721
        self.target_width = 1440
        self.scale_factor = scale_factor
        
        self.inc = DoubleConv(in_chans + 1, hidden_dim)
        self.down1 = DownWithResidual(hidden_dim, hidden_dim * 2, n_layers=1)
        self.down2 = DownWithResidual(hidden_dim * 2, hidden_dim * 4, n_layers=1) 
        self.down3 = DownWithResidual(hidden_dim * 4, hidden_dim * 8, n_layers=1)
        
        self.up1 = UpWithSkip(hidden_dim * 8, hidden_dim * 4)
        self.up2 = UpWithSkip(hidden_dim * 4, hidden_dim * 2)
        self.up3 = UpWithSkip(hidden_dim * 2, hidden_dim)
        
        self.out_conv = nn.Conv2d(hidden_dim, out_chans, kernel_size=1)
        self.residual_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        
        self.use_checkpoint = True

    def forward(self, x, topography=None):
        x_residual = x
        
        if topography is None:
            topography = torch.zeros(x.size(0), 1, x.size(2), x.size(3), 
                                   device=x.device, dtype=x.dtype)
        
        x = torch.cat([x, topography], dim=1)
        
        if self.use_checkpoint and self.training:
            x1 = checkpoint(self.inc, x)
            x2 = checkpoint(self.down1, x1)
            x3 = checkpoint(self.down2, x2)
            x4 = checkpoint(self.down3, x3)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2) 
            x4 = self.down3(x3)
        
        if self.use_checkpoint and self.training:
            x = checkpoint(self.up1, x4, x3)
            x = checkpoint(self.up2, x, x2)
            x = checkpoint(self.up3, x, x1)
        else:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)  
            x = self.up3(x, x1)
        
        x = self.out_conv(x)
        
        x = F.interpolate(
            x, 
            size=(self.target_height, self.target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        x_residual_upscaled = F.interpolate(
            x_residual, 
            size=(self.target_height, self.target_width), 
            mode='bilinear', 
            align_corners=False
        )
        x_residual_upscaled = self.residual_conv(x_residual_upscaled)
        
        return x + x_residual_upscaled

def create_weather_unet(model_type='unet', **kwargs):
    if model_type == 'unet':
        return WeatherUNet(**kwargs)
    

def load_trained_model(checkpoint_path, model_class, model_kwargs, device='cpu'):
    """Загрузка обученной модели из checkpoint"""
    print(f"Загрузка модели из {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = model_class(**model_kwargs)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Загружен model_state_dict из checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Загружен state_dict из checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Загружен checkpoint как state_dict")
        
        model = model.to(device)
        model.eval()  
        
        print(f"Модель загружена и перемещена на {device}")
        print(f"Архитектура: {model.__class__.__name__}")
        
        return model
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        raise