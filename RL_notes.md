# Aprendizado por Reforço

- Conceitos chaves:
  
  - Agentes, Ambiente, Politica de Recompensa, Fatores de Desconto e a Equação de Bellman.
  
  - Processo Determinístico vs. Estocástico.
  
  - Aprendizado por Reforço Tabular.

- Posteriormente tópicos sobre Aprendizado por Reforço Profundo.

## Introdução

Terminologias: Agentes, Ambientes, Estados, Observações e Politica.

> Muito frequentemente as variáveis utilizadas não apresentaram os mesmo nomes associados as convenções das terminologias.

Relação estado vs. observação:

> Um estado "s" é uma completa descrição do estado daquele mundo, não existindo qualquer informação desse mundo escondida desse estado.

> Um observação "o" é uma descrição parcial do estado, podendo omitir informações presentes naquele mundo.

> Na maioria dos ambientes, não teremos todas informações para compor um estado verdadeiro. Porém, muitas dessas informações desconhecidas podem não ser uteis para a representação daquele ambiente.

## OpenAI Gym

A biblioteca OpenAI Gym é umas das ferramentas chaves para aprender como criar algoritmos de Aprendizado de Máquina por Reforço.

Os objetivos para utiliza-la são: 

- Conhecer como abrir um ambiente gym.

- Interagir com ações de um agente do gym baseado em observações do ambiente.

Maioria do ambientes Gym são focados em "games", porém iremos é possível adaptá-los para funcionar em qualquer programa python ou entrada visual.

Games provem um framework bem esclarecido em relação de definição de ambiente, observações, agentes e leque de ações.



Maioria dos ambientes possuem duas versões:

- RAM: Apenas retorna algumas propriedades como coordenada da bola, localização da plataforma.
  
  - Muito útil para ambientes simples que não precisam "enxergar"" para treinamento.

- Versão Padrão/Imagem: Retornando um histórico de imagens.
  
  - Ambiente mais realístico para treinamento, mas irá necessitar de redes mais sofisticadas como CNN.

Iremos explorar o ambiente do game Breakout, e seus conceitos chaves para criação de um ambiente.

Posteriormente, iremos também explorar interações com ambiente com um agente muito simplório que irá escolher ações aleatórias.
