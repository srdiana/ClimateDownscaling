import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import stats
from typing import Tuple, List, Optional, Dict
import warnings

def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """Вычисляет среднеквадратичную ошибку (Root Mean Square Error)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray):
    """Вычисляет среднюю абсолютную ошибку (Mean Absolute Error)"""
    return mean_absolute_error(y_true, y_pred)

def r_squared(y_true: np.ndarray, y_pred: np.ndarray):
    """Вычисляет коэффициент детерминации (R-squared)"""
    return r2_score(y_true, y_pred)

def bias(y_true: np.ndarray, y_pred: np.ndarray):
    """Вычисляет смещение"""
    return np.mean(y_pred - y_true)

def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: Optional[float] = None):
    """
    Вычисляет пиковое отношение сигнал/шум (Peak Signal-to-Noise Ratio)
    max_val: максимальное значение динамического диапазона данных
    """
    if max_val is None:
        max_val = np.max(y_true)  
    
    mse = mean_squared_error(y_true, y_pred)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

# def power_spectral_density(signal, fs=1.0):
#     """
#     Вычисляет спектральную плотность мощности (Power Spectral Density)
#     signal: входной сигнал
#     fs: частота дискретизации
#     """
#     frequencies, psd = signal.welch(signal, fs)
#     return frequencies, psd

def power_spectral_density(signal_data: np.ndarray, fs: float = 1.0, 
                          nperseg: int = None, scaling: str = 'density') -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет спектральную плотность мощности (Power Spectral Density)
    
    Args:
        signal_data: входной сигнал (1D или 2D)
        fs: частота дискретизации
        nperseg: длина сегмента для метода Уэлча
        scaling: тип нормировки ('density' или 'spectrum')
        
    Returns:
        frequencies: массив частот
        psd: спектральная плотность мощности
    """
    if signal_data.ndim > 1:
        # Для 2D данных вычисляем PSD по каждому измерению и усредняем
        if nperseg is None:
            nperseg = min(256, signal_data.shape[-1] // 2)
        
        if signal_data.ndim == 2:
            psd_list = []
            for i in range(signal_data.shape[0]):
                f, pxx = signal.welch(signal_data[i], fs=fs, nperseg=nperseg, scaling=scaling)
                psd_list.append(pxx)
            psd = np.mean(psd_list, axis=0)
        else:
            # Для 1D
            f, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg, scaling=scaling)
    else:
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 2)
        f, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg, scaling=scaling)
    
    return f, psd

# def spectral_rmse(y_true, y_pred, fs=1.0):
#     """
#     Вычисляет спектральную RMSE между двумя сигналами
#     fs: частота дискретизации
#     """
#     # Вычисление PSD для обоих сигналов
#     _, psd_true = power_spectral_density(y_true, fs)
#     _, psd_pred = power_spectral_density(y_pred, fs)
    
#     # Обрезка до минимальной длины
#     min_len = min(len(psd_true), len(psd_pred))
#     psd_true = psd_true[:min_len]
#     psd_pred = psd_pred[:min_len]
    
#     return np.sqrt(np.mean((psd_true - psd_pred) ** 2))

def spectral_rmse(y_true: np.ndarray, y_pred: np.ndarray, fs: float = 1.0) -> float:
    """
    Вычисляет спектральную RMSE между двумя сигналами
    
    Args:
        y_true: эталонный сигнал
        y_pred: предсказанный сигнал
        fs: частота дискретизации
        
    Returns:
        spectral_rmse: спектральная RMSE
    """
    _, psd_true = power_spectral_density(y_true, fs)
    _, psd_pred = power_spectral_density(y_pred, fs)
    
    min_len = min(len(psd_true), len(psd_pred))
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]
    
    return np.sqrt(np.mean((psd_true - psd_pred) ** 2))

def correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет коэффициент корреляции Пирсона"""
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

def spectral_mae(y_true: np.ndarray, y_pred: np.ndarray, fs: float = 1.0) -> float:
    """
    Вычисляет спектральную MAE между двумя сигналами
    
    Args:
        y_true: эталонный сигнал
        y_pred: предсказанный сигнал
        fs: частота дискретизации
        
    Returns:
        spectral_mae: спектральная MAE
    """
    # Вычисление PSD для обоих сигналов
    _, psd_true = power_spectral_density(y_true, fs)
    _, psd_pred = power_spectral_density(y_pred, fs)
    
    # Обрезка до минимальной длины
    min_len = min(len(psd_true), len(psd_pred))
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]
    
    return np.mean(np.abs(psd_true - psd_pred))

def spectral_correlation(y_true: np.ndarray, y_pred: np.ndarray, fs: float = 1.0) -> float:
    """
    Вычисляет спектральную корреляцию между двумя сигналами
    
    Args:
        y_true: эталонный сигнал
        y_pred: предсказанный сигнал
        fs: частота дискретизации
        
    Returns:
        spectral_correlation: корреляция между спектрами
    """
    # Вычисление PSD для обоих сигналов
    _, psd_true = power_spectral_density(y_true, fs)
    _, psd_pred = power_spectral_density(y_pred, fs)
    
    # Обрезка до минимальной длины
    min_len = min(len(psd_true), len(psd_pred))
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]
    
    # Вычисление корреляции
    return np.corrcoef(psd_true, psd_pred)[0, 1]

def spectral_bias(y_true: np.ndarray, y_pred: np.ndarray, fs: float = 1.0) -> float:
    """
    Вычисляет спектральное смещение (bias) между двумя сигналами
    
#     Args:
#         y_true: эталонный сигнал
#         y_pred: предсказанный сигнал
#         fs: частота дискретизации
        
#     Returns:
#         spectral_bias: среднее смещение спектров
#     """
    # Вычисление PSD для обоих сигналов
    _, psd_true = power_spectral_density(y_true, fs)
    _, psd_pred = power_spectral_density(y_pred, fs)
    
    # Обрезка до минимальной длины
    min_len = min(len(psd_true), len(psd_pred))
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]
    
    return np.mean(psd_pred - psd_true)

def energy_spectra_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          frequency_bands: List[Tuple[float, float]] = None,
                          fs: float = 1.0) -> Dict[str, float]:
    """
    Вычисляет энергетические метрики в различных частотных полосах
    
    Args:
        y_true: эталонный сигнал
        y_pred: предсказанный сигнал
        frequency_bands: список кортежей с границами частотных полос [(low1, high1), (low2, high2), ...]
        fs: частота дискретизации
        
    Returns:
        metrics_dict: словарь с метриками по частотным полосам
    """
    if frequency_bands is None:
        nyquist = fs / 2
        frequency_bands = [
            (0, nyquist/4),           
            (nyquist/4, nyquist/2),  
            (nyquist/2, nyquist)   
        ]
    
    # Вычисление PSD для обоих сигналов
    freqs, psd_true = power_spectral_density(y_true, fs)
    _, psd_pred = power_spectral_density(y_pred, fs)
    
    # Обрезка до минимальной длины
    min_len = min(len(psd_true), len(psd_pred))
    freqs = freqs[:min_len]
    psd_true = psd_true[:min_len]
    psd_pred = psd_pred[:min_len]
    
    metrics = {}
    
    for i, (f_low, f_high) in enumerate(frequency_bands):
        # Маска для текущей частотной полосы
        band_mask = (freqs >= f_low) & (freqs < f_high)
        
        if np.any(band_mask):
            # Энергия в полосе (интеграл от PSD)
            energy_true = np.trapz(psd_true[band_mask], freqs[band_mask])
            energy_pred = np.trapz(psd_pred[band_mask], freqs[band_mask])
            
            # Вычисление метрик
            rel_error = np.abs(energy_pred - energy_true) / energy_true if energy_true > 0 else np.nan
            
            metrics[f'band_{i}_energy_true'] = energy_true
            metrics[f'band_{i}_energy_pred'] = energy_pred
            metrics[f'band_{i}_relative_error'] = rel_error
            metrics[f'band_{i}_freq_range'] = f"{f_low:.2f}-{f_high:.2f}"
    
    return metrics

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         fs: float = 1.0,
                         max_val: Optional[float] = None) -> Dict[str, float]:
    """
    Вычисляет все метрики (пространственные и спектральные)
    
    Args:
        y_true: эталонный сигнал (может быть 1D, 2D или 3D)
        y_pred: предсказанный сигнал
        fs: частота дискретизации для спектральных метрик
        max_val: максимальное значение для PSNR
        
    Returns:
        all_metrics: словарь со всеми вычисленными метриками
#     """
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Формы массивов не совпадают: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    all_metrics = {}
    
    all_metrics['rmse'] = float(rmse(y_true_flat, y_pred_flat))
    all_metrics['mae'] = float(mae(y_true_flat, y_pred_flat))
    all_metrics['r_squared'] = float(r_squared(y_true_flat, y_pred_flat))
    all_metrics['correlation'] = float(correlation_coefficient(y_true_flat, y_pred_flat))
    all_metrics['bias'] = float(bias(y_true_flat, y_pred_flat))
    
    psnr_val = psnr(y_true_flat, y_pred_flat, max_val)
    all_metrics['psnr'] = float(psnr_val) if not np.isinf(psnr_val) else np.nan
    
    if y_true.ndim >= 2:
        for dim in range(min(y_true.ndim, 3)):  
            axis = tuple([d for d in range(y_true.ndim) if d != dim])
            
            if axis:  
                rmse_dim = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))
                mae_dim = np.mean(np.abs(y_true - y_pred), axis=axis)
                
                all_metrics[f'rmse_dim{dim}_mean'] = float(np.mean(rmse_dim))
                all_metrics[f'rmse_dim{dim}_std'] = float(np.std(rmse_dim))
                all_metrics[f'mae_dim{dim}_mean'] = float(np.mean(mae_dim))
                all_metrics[f'mae_dim{dim}_std'] = float(np.std(mae_dim))
    
   
    try:
        if y_true.ndim == 1:
            
            all_metrics['spectral_rmse'] = float(spectral_rmse(y_true, y_pred, fs))
            all_metrics['spectral_mae'] = float(spectral_mae(y_true, y_pred, fs))
            all_metrics['spectral_correlation'] = float(spectral_correlation(y_true, y_pred, fs))
            all_metrics['spectral_bias'] = float(spectral_bias(y_true, y_pred, fs))
            
        elif y_true.ndim == 2:
            spectral_rmse_vals = []
            spectral_mae_vals = []
            spectral_corr_vals = []
            spectral_bias_vals = []
            
            for i in range(y_true.shape[0]):
                spectral_rmse_vals.append(spectral_rmse(y_true[i], y_pred[i], fs))
                spectral_mae_vals.append(spectral_mae(y_true[i], y_pred[i], fs))
                spectral_corr_vals.append(spectral_correlation(y_true[i], y_pred[i], fs))
                spectral_bias_vals.append(spectral_bias(y_true[i], y_pred[i], fs))
            
            # По столбцам
            for j in range(y_true.shape[1]):
                spectral_rmse_vals.append(spectral_rmse(y_true[:, j], y_pred[:, j], fs))
                spectral_mae_vals.append(spectral_mae(y_true[:, j], y_pred[:, j], fs))
                spectral_corr_vals.append(spectral_correlation(y_true[:, j], y_pred[:, j], fs))
                spectral_bias_vals.append(spectral_bias(y_true[:, j], y_pred[:, j], fs))
            
            all_metrics['spectral_rmse'] = float(np.mean(spectral_rmse_vals))
            all_metrics['spectral_mae'] = float(np.mean(spectral_mae_vals))
            all_metrics['spectral_correlation'] = float(np.mean(spectral_corr_vals))
            all_metrics['spectral_bias'] = float(np.mean(spectral_bias_vals))
            
        elif y_true.ndim == 3:
            spectral_rmse_vals = []
            spectral_mae_vals = []
            spectral_corr_vals = []
            spectral_bias_vals = []
            
            for k in range(y_true.shape[0]):
                for i in range(y_true.shape[1]):
                    spectral_rmse_vals.append(spectral_rmse(y_true[k, i], y_pred[k, i], fs))
                    spectral_mae_vals.append(spectral_mae(y_true[k, i], y_pred[k, i], fs))
                    spectral_corr_vals.append(spectral_correlation(y_true[k, i], y_pred[k, i], fs))
                    spectral_bias_vals.append(spectral_bias(y_true[k, i], y_pred[k, i], fs))
            
            all_metrics['spectral_rmse'] = float(np.mean(spectral_rmse_vals))
            all_metrics['spectral_mae'] = float(np.mean(spectral_mae_vals))
            all_metrics['spectral_correlation'] = float(np.mean(spectral_corr_vals))
            all_metrics['spectral_bias'] = float(np.mean(spectral_bias_vals))
            
    except Exception as e:
        warnings.warn(f"Ошибка при вычислении спектральных метрик: {e}")
        all_metrics['spectral_rmse'] = np.nan
        all_metrics['spectral_mae'] = np.nan
        all_metrics['spectral_correlation'] = np.nan
        all_metrics['spectral_bias'] = np.nan
    
    # 5. Energy Spectra метрики
    try:
        energy_metrics = energy_spectra_metrics(y_true_flat, y_pred_flat, fs=fs)
        all_metrics.update(energy_metrics)
    except Exception as e:
        warnings.warn(f"Ошибка при вычислении энергетических метрик: {e}")
    
    return all_metrics

def calculate_spatial_metrics_map(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Вычисляет пространственные метрики для каждого пикселя (карты метрик)
    
    Args:
        y_true: эталонный сигнал [batch, channels, height, width]
        y_pred: предсказанный сигнал [batch, channels, height, width]
        
    Returns:
        metrics_maps: словарь с картами метрик
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Формы массивов не совпадают: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    metrics_maps = {}
    mse_map = np.mean((y_true - y_pred) ** 2, axis=(0, 1))
    metrics_maps['mse'] = mse_map
    metrics_maps['rmse'] = np.sqrt(mse_map)
    
    mae_map = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    metrics_maps['mae'] = mae_map
    
    bias_map = np.mean(y_pred - y_true, axis=(0, 1))
    metrics_maps['bias'] = bias_map
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error_map = np.where(
            np.abs(y_true) > 1e-10,
            np.abs(y_pred - y_true) / np.abs(y_true),
            0
        )
        rel_error_map = np.mean(rel_error_map, axis=(0, 1))
    metrics_maps['relative_error'] = rel_error_map
    
    return metrics_maps

def save_metrics_to_json(metrics: Dict, filepath: str):
    """
    Сохраняет метрики в JSON файл
    
    Args:
        metrics: словарь с метриками
        filepath: путь к файлу для сохранения
    """
    import json
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
