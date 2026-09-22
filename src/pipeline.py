from dataclasses import dataclass, field

import numpy as np

from .operacoes_pontuais import correcao_gama, equalizacao_histograma
from .ruido import ruido_gaussiano
from .filtros_espaciais import (filtro_gaussiano, filtro_high_boost,
                                filtro_mediana, realce_laplaciano)

SIGMA_RUIDO = 15.0
SEMENTE = 42


@dataclass
# Guarda as etapas
class ResultadoPipeline:
    nome: str
    original: np.ndarray
    ruidosa: np.ndarray
    etapas: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def final(self) -> np.ndarray:
        return list(self.etapas.values())[-1]


# Config A: equalização + mediana + high-boost
def configuracao_a(original: np.ndarray) -> ResultadoPipeline:
    ruidosa = ruido_gaussiano(original, sigma=SIGMA_RUIDO, semente=SEMENTE)
    resultado = ResultadoPipeline("Config A (equaliza+mediana+high-boost)",
                                  original, ruidosa)
    equalizada = equalizacao_histograma(ruidosa)
    resultado.etapas["1. Equalização de histograma"] = equalizada
    suavizada = filtro_mediana(equalizada, tamanho=3)
    resultado.etapas["2. Filtro da mediana 3x3"] = suavizada
    agucada = filtro_high_boost(suavizada, k=1.2, tamanho=5, sigma=1.0)
    resultado.etapas["3. High-boost (k=1.2)"] = agucada
    return resultado


# Config B: gaussiano + gama + laplaciano
def configuracao_b(original: np.ndarray) -> ResultadoPipeline:
    ruidosa = ruido_gaussiano(original, sigma=SIGMA_RUIDO, semente=SEMENTE)
    resultado = ResultadoPipeline("Config B (gaussiano+gama+laplaciano)",
                                  original, ruidosa)
    suavizada = filtro_gaussiano(ruidosa, tamanho=5, sigma=1.0)
    resultado.etapas["1. Filtro gaussiano 5x5 (σ=1)"] = suavizada
    corrigida = correcao_gama(suavizada, gama=0.7)
    resultado.etapas["2. Correção gama (γ=0.7)"] = corrigida
    agucada = realce_laplaciano(corrigida, peso=0.8, diagonais=False)
    resultado.etapas["3. Laplaciano somado (peso=0.8)"] = agucada
    return resultado
