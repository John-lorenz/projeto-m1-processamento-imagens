from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metricas import contraste_rms, psnr, ssim
from src.pipeline import SIGMA_RUIDO, configuracao_a, configuracao_b
from src.util import carregar_cinza, histograma_manual, salvar

PASTA_IMAGENS = Path(__file__).parent / "imagens"
PASTA_RESULTADOS = Path(__file__).parent / "resultados"


# Processa uma imagem nas duas configs
def processar_imagem(caminho: Path) -> list[dict]:
    nome = caminho.stem
    original = carregar_cinza(caminho)
    linhas_metricas = []
    resultados = [configuracao_a(original), configuracao_b(original)]

    ruidosa = resultados[0].ruidosa
    linhas_metricas.append({
        "imagem": nome, "config": "-", "etapa": f"Ruidosa (sigma={SIGMA_RUIDO:g})",
        "psnr": psnr(original, ruidosa), "ssim": ssim(original, ruidosa),
        "contraste": contraste_rms(ruidosa),
    })
    salvar(ruidosa, PASTA_RESULTADOS / nome / "00_ruidosa.png")
    salvar(original, PASTA_RESULTADOS / nome / "00_original.png")

    for rotulo, resultado in zip(["A", "B"], resultados):
        for i, (etapa, img) in enumerate(resultado.etapas.items(), start=1):
            arquivo = f"{rotulo}_{i}_{etapa.split('. ', 1)[1][:30]}.png"
            arquivo = (arquivo.replace(" ", "_").replace("=", "")
                       .replace("γ", "gama").replace("σ", "sigma"))
            salvar(img, PASTA_RESULTADOS / nome / arquivo)

        final = resultado.final
        linhas_metricas.append({
            "imagem": nome, "config": rotulo, "etapa": "Final",
            "psnr": psnr(original, final), "ssim": ssim(original, final),
            "contraste": contraste_rms(final),
        })
        gerar_figura_pipeline(nome, rotulo, resultado)

    gerar_figura_histogramas(nome, original, resultados)
    return linhas_metricas


# Figura das etapas
def gerar_figura_pipeline(nome, rotulo, resultado):
    imagens = [("Original", resultado.original),
               (f"Ruidosa (σ={SIGMA_RUIDO:g})", resultado.ruidosa)]
    imagens += list(resultado.etapas.items())

    fig, eixos = plt.subplots(1, len(imagens), figsize=(4 * len(imagens), 4.6))
    fig.suptitle(f"{nome} — {resultado.nome}", fontsize=13)
    for eixo, (titulo, img) in zip(eixos, imagens):
        eixo.imshow(img, cmap="gray", vmin=0, vmax=255)
        eixo.set_title(titulo, fontsize=9)
        eixo.axis("off")
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / nome / f"figura_pipeline_{rotulo}.png", dpi=130)
    plt.close(fig)


# Figura dos histogramas
def gerar_figura_histogramas(nome, original, resultados):
    series = [("Original", original),
              ("Ruidosa", resultados[0].ruidosa),
              ("Final A", resultados[0].final),
              ("Final B", resultados[1].final)]
    fig, eixos = plt.subplots(1, 4, figsize=(16, 3.4))
    fig.suptitle(f"{nome} — Histogramas", fontsize=13)
    for eixo, (titulo, img) in zip(eixos, series):
        eixo.bar(range(256), histograma_manual(img), width=1.0, color="#444")
        eixo.set_title(titulo, fontsize=10)
        eixo.set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(PASTA_RESULTADOS / nome / "figura_histogramas.png", dpi=130)
    plt.close(fig)


# Tabela PSNR / SSIM
def imprimir_tabela(linhas):
    cab = f"{'Imagem':<26}{'Config':<8}{'Etapa':<22}{'PSNR (dB)':>10}{'SSIM':>8}{'Contraste':>11}"
    print("\n" + cab)
    print("-" * len(cab))
    texto = [cab, "-" * len(cab)]
    for l in linhas:
        linha = (f"{l['imagem']:<26}{l['config']:<8}{l['etapa']:<22}"
                 f"{l['psnr']:>10.2f}{l['ssim']:>8.4f}{l['contraste']:>11.2f}")
        print(linha)
        texto.append(linha)
    (PASTA_RESULTADOS / "metricas.txt").write_text("\n".join(texto), encoding="utf-8")


# Entrada do programa
def main():
    extensoes = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    arquivos = sorted(a for ext in extensoes for a in PASTA_IMAGENS.glob(ext))
    if not arquivos:
        raise FileNotFoundError(str(PASTA_IMAGENS))

    todas_metricas = []
    for arquivo in arquivos:
        todas_metricas.extend(processar_imagem(arquivo))
    imprimir_tabela(todas_metricas)


if __name__ == "__main__":
    main()
