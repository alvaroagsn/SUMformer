# SUMformer: Aprimoramento e Avaliação

Este repositório contém uma implementação do modelo **SUMformer**, conforme proposto no artigo *SUMformer: A Spatially-Uniform-Attention-based Model for Efficient Long-Term Spatio-Temporal Forecasting*.

O foco deste projeto foi ir além da replicação, implementando e avaliando uma série de modificações estratégicas para aprimorar o desempenho do modelo na tarefa de previsão de fluxo de tráfego urbano.

## Visão Geral do Projeto

O SUMformer é um modelo baseado em Transformer projetado para ser eficiente em previsões de longo prazo em dados espaço-temporais. Sua principal inovação é o uso de um **Time-Varying Filter (TVF)**, que substitui a atenção self-attention por um mecanismo de complexidade linear, tornando-o ideal para datasets de grande escala.

Este trabalho explora o impacto de:

  - **Funções de Perda Alternativas:** Integração da `SharpLoss` (baseada em DTW) para melhor captura da dinâmica temporal.
  - **Variantes Arquitetônicas:** Uso da `Attention-Fourier (AF)` para explorar a sazonalidade dos dados.
  - **Otimização da Regularização:** Ajuste fino da taxa de `drop_path`.

## Pré-requisitos

Antes de executar o projeto, certifique-se de que você tem um ambiente Python configurado (preferencialmente `Python 3.8+`) e as seguintes bibliotecas instaladas.

```bash
pip install torch numpy pandas matplotlib timm
```

## Estrutura do Dataset

Para que o script funcione corretamente, os datasets devem seguir uma estrutura específica. Por exemplo, para o dataset NYC:

1.  Crie uma pasta `datasets/NYC/`.
2.  Dentro dela, coloque os arquivos:
      - `NYC_2015.npy`: Contém os dados de fluxo da grade espaço-temporal.
      - `NYC_timestamp.npy`: Contém os timestamps correspondentes.

A estrutura deve ser similar para os outros datasets (`TaxiBJ`, `Chengdu`).

## Como Executar o Modelo

O script principal para treinar e avaliar o modelo é o `Sumformer_origin_exp_full.py`. Abaixo estão exemplos de comandos para executar os experimentos descritos neste projeto.

### 1\. Treinando o Modelo com a Configuração Otimizada (AF + SharpLoss + DropPath 0.2)

Este é o comando para executar a melhor configuração que encontramos.

```bash
python Sumformer_origin_exp_full.py \
    --device cuda \
    --dataset NYC \
    --pth "pth/SUMformer_AF_NYC_best.pth" \
    --batch 16 \
    --lr 0.00075
```