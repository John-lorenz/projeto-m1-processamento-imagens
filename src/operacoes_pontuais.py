import numpy as np


# Equalização de histograma
def equalizacao_histograma(imagem: np.ndarray) -> np.ndarray:
    img = imagem.astype(np.uint8)

    histograma = np.zeros(256, dtype=np.int64)
    for nivel in range(256):
        histograma[nivel] = np.sum(img == nivel)

    cdf = np.zeros(256, dtype=np.int64)
    acumulado = 0
    for nivel in range(256):
        acumulado += histograma[nivel]
        cdf[nivel] = acumulado

    total_pixels = img.size
    cdf_min = cdf[cdf > 0][0]
    denominador = total_pixels - cdf_min
    if denominador == 0:
        return img.copy()

    mapa = np.round((cdf - cdf_min) / denominador * 255.0)
    mapa = np.clip(mapa, 0, 255).astype(np.uint8)
    return mapa[img]


# Correção gama
def correcao_gama(imagem: np.ndarray, gama: float, c: float = 1.0) -> np.ndarray:
    img = imagem.astype(np.float64) / 255.0
    saida = c * np.power(img, gama)
    saida = np.clip(saida * 255.0, 0, 255)
    return saida.astype(np.uint8)
