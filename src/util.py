from pathlib import Path

import numpy as np
from PIL import Image


# Carrega em cinza
def carregar_cinza(caminho: str | Path) -> np.ndarray:
    with Image.open(caminho) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


# Salva PNG
def salvar(imagem: np.ndarray, caminho: str | Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagem.astype(np.uint8)).save(caminho)


# Histograma
def histograma_manual(imagem: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for nivel in range(256):
        hist[nivel] = np.sum(imagem == nivel)
    return hist
