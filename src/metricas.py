import numpy as np

from .filtros_espaciais import convolucao2d, kernel_gaussiano


# MSE
def mse(referencia: np.ndarray, avaliada: np.ndarray) -> float:
    ref = referencia.astype(np.float64)
    ava = avaliada.astype(np.float64)
    return float(np.mean((ref - ava) ** 2))


# PSNR
def psnr(referencia: np.ndarray, avaliada: np.ndarray,
         valor_maximo: float = 255.0) -> float:
    erro = mse(referencia, avaliada)
    if erro == 0:
        return float("inf")
    return float(10.0 * np.log10((valor_maximo ** 2) / erro))


# SSIM
def ssim(referencia: np.ndarray, avaliada: np.ndarray,
         valor_maximo: float = 255.0) -> float:
    x = referencia.astype(np.float64)
    y = avaliada.astype(np.float64)

    c1 = (0.01 * valor_maximo) ** 2
    c2 = (0.03 * valor_maximo) ** 2
    janela = kernel_gaussiano(11, 1.5)

    mu_x = convolucao2d(x, janela)
    mu_y = convolucao2d(y, janela)
    sigma_x2 = convolucao2d(x * x, janela) - mu_x ** 2
    sigma_y2 = convolucao2d(y * y, janela) - mu_y ** 2
    sigma_xy = convolucao2d(x * y, janela) - mu_x * mu_y

    numerador = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominador = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerador / denominador))


# Contraste RMS
def contraste_rms(imagem: np.ndarray) -> float:
    return float(np.std(imagem.astype(np.float64)))
