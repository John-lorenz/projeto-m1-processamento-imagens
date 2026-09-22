import numpy as np


# Janelas k×k (apoio da convolução e da mediana)
def _janelas_deslizantes(imagem_pad: np.ndarray, k: int) -> np.ndarray:
    alt = imagem_pad.shape[0] - k + 1
    larg = imagem_pad.shape[1] - k + 1
    s0, s1 = imagem_pad.strides
    return np.lib.stride_tricks.as_strided(
        imagem_pad,
        shape=(alt, larg, k, k),
        strides=(s0, s1, s0, s1),
        writeable=False,
    )


# Convolução 2D
def convolucao2d(imagem: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    if kernel.shape[0] != kernel.shape[1] or k % 2 == 0:
        raise ValueError("Kernel deve ser quadrado e de tamanho ímpar.")

    kernel_rot = kernel[::-1, ::-1].astype(np.float64)
    borda = k // 2
    img = imagem.astype(np.float64)
    img_pad = np.pad(img, borda, mode="edge")
    janelas = _janelas_deslizantes(img_pad, k)
    return np.einsum("xyij,ij->xy", janelas, kernel_rot)


# Kernel gaussiano
def kernel_gaussiano(tamanho: int = 5, sigma: float = 1.0) -> np.ndarray:
    borda = tamanho // 2
    eixo = np.arange(-borda, borda + 1, dtype=np.float64)
    xx, yy = np.meshgrid(eixo, eixo)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# Filtro gaussiano
def filtro_gaussiano(imagem: np.ndarray, tamanho: int = 5,
                     sigma: float = 1.0) -> np.ndarray:
    saida = convolucao2d(imagem, kernel_gaussiano(tamanho, sigma))
    return np.clip(saida, 0, 255).astype(np.uint8)


# Filtro da mediana
def filtro_mediana(imagem: np.ndarray, tamanho: int = 3) -> np.ndarray:
    if tamanho % 2 == 0:
        raise ValueError("Tamanho da janela deve ser ímpar.")
    borda = tamanho // 2
    img_pad = np.pad(imagem.astype(np.float64), borda, mode="edge")
    janelas = _janelas_deslizantes(img_pad, tamanho)
    alt, larg = imagem.shape
    planas = janelas.reshape(alt, larg, tamanho * tamanho)
    ordenadas = np.sort(planas, axis=2)
    mediana = ordenadas[:, :, (tamanho * tamanho) // 2]
    return np.clip(mediana, 0, 255).astype(np.uint8)


KERNEL_LAPLACIANO_4 = np.array([[0, -1,  0],
                                [-1, 4, -1],
                                [0, -1,  0]], dtype=np.float64)

KERNEL_LAPLACIANO_8 = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float64)


# Laplaciano somado à imagem
def realce_laplaciano(imagem: np.ndarray, peso: float = 1.0,
                      diagonais: bool = False) -> np.ndarray:
    kernel = KERNEL_LAPLACIANO_8 if diagonais else KERNEL_LAPLACIANO_4
    lap = convolucao2d(imagem, kernel)
    saida = imagem.astype(np.float64) + peso * lap
    return np.clip(saida, 0, 255).astype(np.uint8)


# High-boost
def filtro_high_boost(imagem: np.ndarray, k: float = 1.5,
                      tamanho: int = 5, sigma: float = 1.0) -> np.ndarray:
    img = imagem.astype(np.float64)
    suave = convolucao2d(imagem, kernel_gaussiano(tamanho, sigma))
    mascara = img - suave
    saida = img + k * mascara
    return np.clip(saida, 0, 255).astype(np.uint8)
