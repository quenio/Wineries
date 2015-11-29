# IA - Trabalho Prático 5 - Neural Networks - Identificação da Vinícula

- Aluno: Quenio Cesar Machado dos Santos
- Matrícula: 14100868
- Semestre: 2015-2

## Introdução

O reconhecimento de padrões através de redes neurais é demonstrado neste trabalho. Para tanto, usaremos dados de análise química de vinhos coletados de três vinhedos. O objetivo é descobrir, dado sua análise química, de qual vinhedo pertence uma amostra de vinho.

O `MatLab`, com seu o `toolbox` de `Neural Networks`, foi a ferramenta utilizada para o tratamento dos dados, treinamento da rede neural e verificação do seu desempenho.

Os dados foram fornecidos em duas matrizes carregáveis no `MatLab`:

- `x`: que contém os dados da análise química dos vinhos, onde cada linha representa o componente químico analisado e cada coluna representa uma amostra de vinho.
- 't': contendo a informação de qual vinhedo pertence a amostra de vinho, onde cada linha representa um vinhedo e cada coluna representa a amostra de vinho correspondendo à coluna de mesmo número na matrix `x`.

## Normalização dos Dados

Antes de definir e treinar a rede neural, é preciso normalizar os dados das análises químicas no intervalo [-1, 1]. A normalização permite o melhor desempenho da função de treinamento da rede quando se usa a função de transferência `tangente sigmoidal`, que tem maior variação no intervalo [-1, 1].

Para a normalização, utilizamos a seguinte função do `Neural Net Toolbox`:

```python
xn = mapminmax(x);
```

`mapminmax` recebe como entra a matriz de dados original e retorna uma matriz de mesma dimensão com os dados normalizados no intervalo [-1, 1].

Esta normalização foi feita nos scripts `quick_net.m` e `verified_net.m` que serão usados nas próximas seções.

## Protótipação Rápida

## Conjuntos de Treinamento e de Teste

Antes de definir a rede neural que irá reconhecer a origem das amostras de vinho, é preciso separar os dados em dois conjuntos distintos:

- _conjunto de treinamento_: que contém os dados que fazem o treinamento da rede neural;
- _conjunto de teste_: contendo os dados que verificam o desenpenho da rede.

Estes conjuntos precisam ser distintos para provar que o treinamento efetuado permite a rede reconhecer novos padrões.

Também é preciso definir o tamanho da fatia do conjunto original de dados que será usado para treinamento e o conjunto que será usado para teste. Para nossos experiementos, usamos as seguintes proporções:

- _dois terços para treinamento_: a maior parte das amostras de vinho foi usada para treinamento, pois uma rede bem treinada deve ter um melhor desempenho na determinação da origem do vinho.
- _um terço para testes_: a menor parte dos dados foi usada para testes, pois é possível verificar o desempenho da rede com um número menor dos amostras, desde que sejam representativas da população de amostras de vinho.
