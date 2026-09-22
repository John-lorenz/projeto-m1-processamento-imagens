# Relatório — Projeto M1: Operações Pontuais e Filtragem Espacial

Universidade do Vale do Itajaí — Ciência da Computação
Processamento de Imagens — Prof. Felipe Viel
João Arthur dos Santos Lorenzoni e Lucas Francelino
22/09/2026

## Enunciado

Neste trabalho usamos radiografias do conjunto MURA e tentamos melhorar a qualidade delas com operação pontual e filtro espacial. Cada etapa foi implementada por nós, sem filtro pronto de biblioteca. No final comparamos duas formas de montar o pipeline, olhando a imagem e também as métricas.

## Por que isso importa

A ideia do professor é que essas imagens iriam para uma análise depois, tipo classificação. Só que raio-X não vem padronizado. Tem exame escuro, tem exame com pouco contraste, tem exame com sujeira. Se o filtro for aplicado errado, o osso some ou o ruído aumenta. Então o ponto não era “aplicar filtro e pronto”, era ver o que realmente ajuda e o que estraga.

## Imagens utilizadas

Ficamos com antebraço (`XR_FOREARM` do MURA v1.1). Escolhemos essa região porque no mesmo exame aparecem rádio e ulna, e a espessura do tecido muda ao longo do braço. Isso já dá variação de contraste sem precisar misturar categorias.

Pegamos três imagens da pasta `valid`, cada uma de um paciente. Se fosse o mesmo paciente, os três recortes iam parecer iguais e não teria o que comparar.

| | Arquivo | De onde saiu | O que observamos |
|---|---|---|---|
| 1 | `antebraco_artefatos.png` | patient11328, study1_positive, image1 | Tem a mão junto, lençol e aquele marcador por cima |
| 2 | `antebraco_bem_definido.png` | patient11362, study1_negative, image2 | AP normal, osso mais fácil de ver |
| 3 | `antebraco_subexposto.png` | patient11355, study1_positive, image2 | Lateral escura, contraste fraco |

Olhando o histograma, as três empurram a maior parte dos pixels para o preto (é o fundo da radiografia). A terceira é a pior nesse sentido: quase não tem cinza no meio.

## Implementação

A sequência foi: imagem original, adição de ruído, processamento e comparação com a original. As funções estão em `src/`. Para rodar tudo: `python main.py`.

Colocamos ruído gaussiano com σ = 15 e semente 42. A semente fixa é para as duas configs receberem o mesmo ruído. Dá para ver o granulado no fundo, mas ainda dá para achar o osso.

Não usamos `filter2D` nem equalize de biblioteca. O NumPy só segura a matriz. Pillow abre e salva o arquivo. O Matplotlib serviu só para montar as figuras lado a lado.

As operações:

- equalização de histograma (contagem de cada nível, CDF e mapa);
- correção gama, `s = c * r^γ`, com γ = 0,7;
- convolução 2D, com padding na borda e o kernel virado;
- gaussiano 5×5 (σ = 1) e mediana 3×3;
- high-boost, `g = f + k(f - fsuave)`, k = 1,2;
- laplaciano somado na imagem, `g = f + 0,8 * ∇²f`. Isso importa: o enunciado não quer só a imagem de borda.

### Config A e Config B

| | A | B |
|---|---|---|
| ruído | σ = 15, semente 42 | o mesmo |
| primeiro | equalização | gaussiano 5×5 |
| depois | mediana 3×3 | gama 0,7 |
| por último | high-boost | laplaciano somado |

Trocamos a técnica e também a ordem. Na B o filtro vai antes da gama. Motivo simples: γ menor que 1 clareia o escuro, e o ruído gaussiano incomoda mais no fundo preto. Se equalizar primeiro, como na A, esse ruído sobe junto e depois a mediana não dá conta.

## Código

Equalização, em `src/operacoes_pontuais.py`:

```python
histograma = np.zeros(256, dtype=np.int64)
for nivel in range(256):
    histograma[nivel] = np.sum(img == nivel)

cdf = np.zeros(256, dtype=np.int64)
acumulado = 0
for nivel in range(256):
    acumulado += histograma[nivel]
    cdf[nivel] = acumulado

mapa = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255.0)
return mapa[img]
```

Aguçamento, em `src/filtros_espaciais.py`:

```python
saida = imagem + peso * convolucao2d(imagem, kernel_laplaciano)

mascara = imagem - suave
saida = imagem + k * mascara
```

O pipeline em si está em `src/pipeline.py`. PSNR e SSIM também foram implementados por nós, em `src/metricas.py`.

## Métricas

Usamos PSNR e SSIM, as duas contra a original (antes de sujar com ruído).

PSNR olha o erro de cada pixel. Se o ruído caiu, o número sobe.

SSIM olha se a estrutura continuou parecida. Faz mais sentido em raio-X, porque o que importa é o osso continuar reconhecível, não o cinza ser idêntico.

O contraste RMS só foi anotado para não se enganar: a Config A sempre “ganha” nele, porque a equalização estica o histograma. Só que ela estica o ruído também.

## Resultados

Saída do `python main.py`. As figuras de cada etapa estão em `resultados/`.

| Imagem | Etapa | PSNR (dB) | SSIM | Contraste |
|---|---|---|---|---|
| artefatos | ruidosa | 24,97 | 0,3997 | 37,51 |
| artefatos | A | 8,56 | 0,1375 | 73,45 |
| artefatos | B | 19,38 | 0,4564 | 38,60 |
| bem definido | ruidosa | 26,04 | 0,3223 | 33,80 |
| bem definido | A | 9,38 | 0,1064 | 85,23 |
| bem definido | B | 20,29 | 0,3073 | 40,49 |
| subexposto | ruidosa | 24,76 | 0,2868 | 24,82 |
| subexposto | A | 8,94 | 0,1090 | 73,40 |
| subexposto | B | 18,35 | 0,4583 | 25,80 |

Em PSNR e SSIM a B ganhou nas três. A A só ganha no contraste, e olhando a figura dá para ver por quê: a imagem fica clara e ao mesmo tempo cheia de granulado.

## Análise

Na imagem com artefatos a B foi melhor. O SSIM até passou o da ruidosa (0,46 contra 0,40). O gaussiano tirou um pouco da sujeira e a gama não explodiu o fundo. Na A o lençol e o fundo viraram chuvisco, e o high-boost piorou.

Na AP bem definida também ficou a B. Essa já estava boa, então equalizar não tinha muito o que resolver. O PSNR da B deu uns 20 dB; o da A, 9 dB. O SSIM da B quase empata com a ruidosa. Na A o osso clareia demais e o fundo suja.

A lateral escura foi onde a B mais ajudou. O SSIM saiu de 0,29 na ruidosa para 0,46. A gama clareou o osso sem jogar o ruído do fundo para o meio da escala. Na A o contraste RMS foi para 73 e o SSIM caiu para 0,11. Visualmente parece outra imagem, pior.

O que não deu certo:

Equalizar com fundo quase preto. O histograma é praticamente um pico no zero, então o ruído do fundo sobe junto.

Passar mediana 3×3 depois. Já era tarde. A equalização já tinha espalhado tudo.

High-boost em cima de ruído. A máscara que deveria ser detalhe pega granulado também.

O laplaciano da B também puxa um pouco de ruído, mas bem menos. Dá para continuar lendo o osso.

No geral, a escolha fica com a Config B nas três imagens. A A parece mais “processada”, mas piora o resultado.

Um cuidado com as métricas: o PSNR cai quando o histograma é alterado de propósito, mesmo se o resultado ficar melhor de ver. E o SSIM conta o fundo preto inteiro, não só o braço. Por isso não consideramos apenas o número.

Deixamos os parâmetros iguais nas três imagens. Se fosse ajustar cada uma, não dava para falar que A e B foram comparadas no mesmo cenário.

## Código e resultados

O código acompanha o relatório. As radiografias estão em `imagens/` e o que o programa gera fica em `resultados/`.
