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

## Normalização dos Dados de Entrada

Antes de definir e treinar a rede neural, é preciso normalizar os dados das análises químicas no intervalo [-1, 1]. A normalização permite o melhor desempenho da função de treinamento da rede quando se usa a função de transferência `tangente sigmoidal`, que tem maior variação no intervalo [-1, 1].

Para a normalização, utilizamos a seguinte função do `Neural Net Toolbox`:

```python
xn = mapminmax(x);
```

`mapminmax` recebe como entra a matriz de dados original e retorna uma matriz de mesma dimensão com os dados normalizados no intervalo [-1, 1].

Esta normalização foi feita nos scripts `quick_net.m` e `verified_net.m` que serão usados nas próximas seções.

## Saturação dos Dados de Saída

Na matriz `t`, que representada a saída esperada, tem-se o valor `1` para o vinhedo que é a origem de uma amostra de vinho e o valor `0` para os demais vinhedos. No treinamento da rede neural, a função de treinamento vai ajustar a rede para que esta se aproxime dos valores encontrados em `t`. Como é apenas uma aproximação, precisamos de uma função que converta os valores de saída em zeros e uns. Isto nos permitirá comparar as saídas da rede com os valores em `t` a fim the verificar o desempenho da rede. Este processo é chamado de "saturação".

A função de saturação é mostrada abaixo:

```python
function s = saturate(output)
    row_size = size(output, 1);
    col_size = size(output, 2);
    s = zeros(row_size, col_size);
    for i = 1:col_size
        v = output(:, i);
        if v(1) > v(2) && v(1) > v(3)
            s(1, i) = 1;
        elseif v(2) > v(1) && v(2) > v(3)
            s(2, i) = 1;
        else
            s(3, i) = 1;
        end    
    end
end
```

Observe no código acima que criamos uma nova matriz de mesma dimensão que a matriz de saída (`t`). Inicialmente, esta matriz tem zeros em todas as suas célculas. Depois o valor `1` é colocado na célula da linha que contém o maior valor em `t`, ou seja, o valor mais perto de `1`.

Esta função é utiliza pelos scripts `quick_net.m` e `verified_net.m` - usados nas próximas seções - para "saturar" a saída da rede e permitir a comparação com `t`.

## Protótipação Rápida

A fim de nos familiarizarmos com as funções da `Neural Net Toolbox`, e com o processo de treinamento e verificação, implementamos um script chamado `quick_net.m` que cria, treina e valida uma rede usando o mesmo conjunto de dados.

Veja o código abaixo:

```python
function quick_net(net_size, x, t)
    xn = mapminmax(x);

    net = newff(xn, t, net_size, {'tansig','tansig'}, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 0;

    net = train(net,xn,t);
    y = saturate(sim(net, xn));
    plotconfusion(t, y);
end
```

Na função `quick_net` listada acima, observe o seguinte:

- Além do conjunto de dados de entrada `x` e do conjunto alvo `t`, a função também recebe como entrada o número de neurônios da camada intermediária.
- Primeiramente, a função normaliza os dados usando `mapminmax`.
- Logo após, a rede é criada usando a função de transferência `tangente sigmoidal` tanto para os neurônios da camada intermediária, quanto para os neurônios da camada de saída. Fizemos assim a rede ser compatível com os dados normalizados do conjunto de entrada.
- O treinamento terá a seguinte configuração:
    - 'trainlm': Levenberg-Marquardt é usado no treinamento para que se alcance mínimo do gradiente no menor número de épocas possível.
    - `epochs = 1000`: Mil épocas é mais que suficiente para Levenberg-Marquardt.
    - `goal = 0`: Com erro zero desejamos atingir o mínimo da função, ao invés de apenas uma aproximação.
    - a proporcão de treinamento, validação e teste: como não foram especifados os conjuntos de treinamento, validação e de teste, estes serão aleatóriamente construídos numa proporção de 60%, 20% e 20%, respectivamente.
- Uma vez treinada a rede, a função `quick_net`, executa a rede sobre os mesmos dados de treinamento e compara o resultado com os dados originais num gráfico. Esta não é a melhor de verificar a rede, como veremos nas próximas seções, mas é suficiente para nossa prototipação rápida.

Para fazer experimentos com o script acima, criou-se redes de 3 a 7 neurônios intermediários sobre todas as treze análises químicas fornecidas no conjunto de dados original. Após várias execuções, o resultado da melhor performance de cada rede foi colocado na pasta `testes` em arquivos com o prefixo `quick_13i_`.

Observamos nos experimentos que foi possível gerar redes que alcançam 100% de acertos em todas as configurações - de 3 a 7 neurônios - como demonstrado nos arquivos da pasta `testes`. Porém, nem todas a redes geradas atingiram este nível de acerto. A taxa de acerto variava entre 95% a 100%.

Como já dissemos antes, não é recomandado usar o mesmo conjunto de treinamento para os testes. A próxima seção cuidará desta questão.

## Conjuntos de Treinamento e de Teste

Usando o mesmo conjunto para treinamento e para a verificação da rede neural, como foi feito na seção anterior de prototipação, não vai mostrar como a rede se comporta com novas amostras.

Portanto, antes de definir a rede neural que irá reconhecer a origem das amostras de vinho, é preciso separar os dados em dois conjuntos distintos:

- _conjunto de treinamento_: que contém os dados que fazem o treinamento da rede neural;
- _conjunto de teste_: contendo os dados que verificam o desempenho da rede.

Também é preciso definir o tamanho da fatia do conjunto original de dados que será usado para treinamento e o tamanho da fatia que será usada para teste. Para nossos experimentos, usamos as seguintes proporções:

- _dois terços para treinamento_: a maior parte das amostras de vinho foi usada para treinamento, pois uma rede bem treinada deve ter um melhor desempenho na determinação da origem do vinho.
- _um terço para testes_: a menor parte dos dados foi usada para testes, pois é possível verificar o desempenho da rede com um número menor dos amostras, desde que sejam representativas da população de amostras de vinho.

Como os dados de amostra foram ordenados por vinhedo, usou-se o código seguinte para distribuir igualmente os vinhedos entre os conjuntos de treinamento e teste:

```python
x_train = [xn(:,1:3:end), xn(:,3:3:end)];
x_test = xn(:,2:3:end);

t_train = [t(:,1:3:end), t(:,3:3:end)];
t_test = t(:,2:3:end);
```

Observe que no código acima intercalamos entre as colunas para conseguir uma variedade entre os conjuntos na proporção desejada.

## Arquitetura da Rede Neural
