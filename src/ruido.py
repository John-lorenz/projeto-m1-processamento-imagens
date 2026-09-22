import numpy as np


# Ruído gaussiano
def ruido_gaussiano(imagem: np.ndarray, sigma: float = 15.0,
                    semente: int = 42) -> np.ndarray:
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0.0, sigma, size=imagem.shape)
    saida = imagem.astype(np.float64) + ruido
    return np.clip(saida, 0, 255).astype(np.uint8)
