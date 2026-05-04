Tanto o PyTorch quanto o TensorFlow cumprem o mesmo objetivo primário no ecossistema de Deep Learning: fornecer infraestrutura para manipulação de tensores em hardware acelerado (GPUs/TPUs) e a computação de gradientes através de diferenciação automática (*autograd*).

 A distinção central entre as ferramentas reside no paradigma arquitetural e no foco estratégico de mercado (pesquisa versus produção industrial). 

### **PyTorch: Flexibilidade e Grafos Dinâmicos**

*Se o objetivo é criar uma arquitetura inédita, manipular dinamicamente o fluxo de dados ou realizar prototipagem rápida, o PyTorch oferece a menor fricção intelectual.* 

Desenvolvido pela Meta (antigo Facebook AI Research), o PyTorch opera sob o paradigma de **Grafos Computacionais Dinâmicos** (*Define-by-Run* ou *Eager Execution* nativo). Isso significa que o grafo da rede neural é construído em tempo de execução, linha a linha.

É o padrão ouro atual na pesquisa acadêmica e no desenvolvimento de modelos de ponta.

A arquitetura dinâmica permite que o código seja estritamente "Pythonico", facilitando o uso de ferramentas nativas do Python para depuração linha a linha de tensores complexos e loops customizados. 

### **TensorFlow: Ecossistema de Produção e Grafos Estáticos**

A força do TensorFlow não está apenas no treinamento, mas na esteira de *deployment*. Ele possui um ecossistema maduro focado em engenharia de software de grande escala. Ferramentas como o **TensorFlow Serving** (para APIs de inferência robustas), **TensorFlow Lite** (para dispositivos móveis e IoT) e **TensorFlow.js** formam uma suíte que resolve o ciclo de vida completo do modelo. É a escolha lógica para sistemas corporativos onde o modelo precisa ser serializado e implantado em hardwares heterogêneos. 

Geralmente acessado através de sua API de alto nível (Keras), o TensorFlow favorece uma declaração mais funcional e sequencial, abstraindo a matemática subjacente para acelerar o desenvolvimento de modelos maduros. 

Em suma, a transição entre ambos é fluida quando os conceitos matemáticos subjacentes estão dominados. A decisão de engenharia deve se basear no destino do projeto: PyTorch para P\&D, flexibilidade algorítmica e controle granular; TensorFlow para arquiteturas padronizadas e deploy agressivo em múltiplos ambientes corporativos 

**Aviso\!**

Utilizar **ambos,** PyTorch e TensorFlow na mesma aplicação não é o padrão da indústria para modelos simples, pois gera redundância e sobrecarga de dependências   
**Pipelines Híbridos de Machine Learning** 

*“…Onde extraímos o melhor dos dois mundos”*

O padrão ouro aqui é:

* Usar o **PyTorch** para a fase de pesquisa, treinamento e extração de características (devido à sua flexibilidade)

   

* O ecossistema **TensorFlow** (como o TF Serving ou TF Lite) para o *deployment* e inferência em alta escala, utilizando protocolos como o ONNX para a conversão entre os frameworks. 

*(**Explicação:** Imagine que você treinou uma inteligência artificial (IA) superinteligente para reconhecer fotos de cachorros no seu computador. O **TensorFlow** é uma ferramenta (framework) para criar modelos de IA.* 

***TF Serving (Servindo):** É o garçom de alta velocidade. Quando sua IA está pronta, você usa o TF Serving para colocá-la em um servidor (nuvem) para que aplicativos da web ou celulares enviem fotos e recebam respostas instantâneas, lidando com milhões de acessos.* 

***TF Lite (Leve):** É a versão "compacta" para dispositivos móveis. Ele pega aquele modelo "pesado" que você treinou e o comprime para rodar dentro de um celular Android, iOS ou até numa geladeira inteligente (IoT), sem precisar de internet.*

***Inferência em Alta Escala:** Inferência é o momento em que a IA faz a previsão.*  
   
***Alta Escala:** Significa que o sistema é robusto o suficiente para atender milhares ou milhões de inferências por segundo sem travar.* 

***ONNX (Open Neural Network Exchange)**: É um formato padrão de arquivo que permite interoperabilidade entre diferentes frameworks. Exemplo, se você treinou 2 modelos, um no TensorFlow e outro no Pytorch; Para estabelecer a comunicação entre os dois modelos ou usar ambos de forma independente, se converte o modelo para o formato **.onnx**. Agora, qualquer ferramenta que suporte ONNX pode usar seu modelo, facilitando a portabilidade entre nuvem, servidor e mobile. )*

**IDEIA CENTRAL**  
*Previsões de Investimento no Mercado Financeiro*  
*(Stock Options)*

“Devido a interessante concatenação de funções entre pytorch e tensor flow. vamos estruturar um modelo na tematica de investimentos envolvendo as top 7 maiores empresas do brasil como petrobras, vale, Banco do Brasil... até a setima em ordem de grandeza.  
eu preciso que você me ajude a entender qual modelo vai fazer o que especificamente, mas a ideia é que um modelo estude toda a tragetória do gráfico de variação da empresa, para entender por exemplo, ' no ano X teve uma das maiores valorizações da empresa até aquele momento, porque? qual foi o contexto historico que motivou isso?' e o outro modelo se especifica em estudar o fator externo como movimentações internacionais por exemplo, noticias, a causa geratriz por conta daquela variação do grafico, que geraram aquela mudança, seja crescimento ou declinio da ação no mercado.  
tendo posse desse montante de dados  com sua devida linkagem de periodo e variancia do stock estudado. o modelo vai estruturar uma previsão final sobre os proximos 7 dias, 3 meses e 1 ano. Imagine que a previsão desse modelo será usada como teste de investimentos; isso é o investidor separa uma fatia razoavel de aporte para 'testar' o palpite da IA alocando seu capital na previsão feita por ela. A IA em resumo, tenta apontar, nesse top 7 se a ação vai subir ou descer, quantos porcento, quando seria ideal a entrada e saida, nos proximos 7 dias, 3 meses e 1 ano.  
também é importante que haja uma função de re-treinamento de modo que se o trader passa 15 dias sem usar o sistema, quando ele abrir, ele faz o re-treino, entende o mercado atual naquele momento, e recebe a previsão.”

**Estruturação da Ideia**

Para as *"Magníficas 7"* da B3 (historicamente orbitando entre Petrobras, Vale, Itaú Unibanco, Ambev, Banco do Brasil, Bradesco e WEG), a volatilidade é altamente ditada por *commodities* globais e risco político-fiscal. A arquitetura precisa refletir essa assimetria. 

### **O Eixo Quantitativo: A Inércia do Preço (TensorFlow)**

Este módulo é responsável por mapear o histórico cru do ativo (OHLCV: *Open, High, Low, Close, Volume*). O objetivo não é prever o futuro ainda, mas criar um modelo que entenda a "assinatura de movimento" de cada uma das 7 empresas.

O **TensorFlow** é a escolha primária aqui devido à sua infraestrutura de otimização de tensores para séries temporais e ferramentas de produção em escala.

A arquitetura ideal para este módulo não é uma LSTM tradicional, mas uma **TCN (Temporal Convolutional Network)** ou um **Time-Series Transformer**. O modelo irá varrer a janela histórica e identificar anomalias matemáticas: quebras de suporte, picos de volume anormais e divergências de volatilidade. A saída deste modelo não é uma decisão de compra, mas sim um vetor matemático (um *embedding*) que resume o estado puramente técnico do ativo naquele momento X.

### **O Eixo Qualitativo: A Causa Geratriz (PyTorch)**

Enquanto o TF diz *o que* aconteceu e a magnitude, o **PyTorch** entra para extrair o *porquê*. Este módulo requer processamento de linguagem natural (NLP) de altíssimo nível para ler o contexto externo.

O **PyTorch** domina este cenário devido à integração nativa com o ecossistema de transformadores (como o Hugging Face). A arquitetura exigirá um modelo de linguagem focado em finanças (como um FinBERT ou uma variação otimizada do LLaMA), treinado para ler três fluxos de dados textuais com *timestamps* exatos:

1. **Macro Internacional:** Preço do barril Brent, minério de ferro em Dalian, curva de juros dos EUA (Fed Funds).  
2. **Macro Nacional:** Decisões do Copom, relatórios do Focus, pautas fiscais.  
3. **Micro (Empresa):** Fatos relevantes da CVM, trocas de CEO, balanços trimestrais.

O **PyTorch** cruzará o *timestamp* da anomalia gráfica detectada pelo TensorFlow com as notícias daquela exata semana. Ele mapeia a semântica causal: "A queda de 8% da PETR4 não foi apenas um ajuste técnico, mas uma reação à declaração governamental sobre política de dividendos cruzada com uma queda no Brent". 

O **PyTorch** gera um segundo *embedding*, representando o "estado fundamental e de sentimento" do ativo.

### **O Eixo de Fusão e Inferência (A Tomada de Decisão)**

O coração do sistema é o módulo de fusão (que pode ser orquestrado em PyTorch por sua facilidade com grafos customizados). Ele concatena o vetor técnico (TensorFlow) com o vetor causal (PyTorch) e passa por uma rede neural final para gerar os três horizontes de previsão solicitados:

* **Curto Prazo (7 dias):** Pesa fortemente o vetor técnico do TF (momentum, fluxo de volume) e o ruído imediato de sentimento. Gera gatilhos táticos de entrada e saída (Swing Trade).  
* **Médio Prazo (3 meses):** O peso se equilibra. O modelo busca antecipar o próximo ciclo de balanços (Earnings Call) cruzado com a tendência de juros.  
* **Longo Prazo (1 ano):** O peso fundamentalista do PyTorch domina. A análise se foca nas políticas de investimento da empresa e superciclos de *commodities*.

A saída da IA não deve ser um palpite absoluto ("Ação vai subir 15%"), mas sim uma distribuição de probabilidade de risco/retorno: *"PETR4: Probabilidade de 68% de upside de 12% em 3 meses. Ponto ótimo de entrada: Retração técnica em R$ X. Stop loss sugerido: R$ Y (suporte histórico mitigado por fluxo de caixa forte)."*

### 

### 

### **A Engenharia do Re-treino Dinâmico (Online Learning)**

A exigência de que o sistema compreenda o mercado atualizado após o usuário ficar 15 dias ausente é o maior desafio de engenharia. Treinar modelos profundos do zero exige vasto poder computacional e tempo, inviabilizando uma atualização rápida ao abrir a plataforma.

A solução é o **Incremental Learning (ou Online Learning)**.

Os modelos base (o leitor de gráficos no TF e o leitor de textos no PyTorch) permanecem com seus pesos principais "congelados" (eles já entendem o que é um suporte ou o que significa a palavra "lucro"). Quando o investidor faz o login após 15 dias, o sistema executa um *pipeline* rápido:

1. Faz o *fetch* dos dados crus dos últimos 15 dias (OHLCV e notícias).  
2. Realiza um *fine-tuning* rápido apenas nas últimas camadas preditivas (as "cabeças" da rede neural), ajustando os pesos exclusivamente para absorver o novo viés de curto prazo.  
3. Isso evita o "esquecimento catastrófico" (onde a rede esquece o histórico de 10 anos ao focar em 15 dias) e permite que a inferência seja gerada em minutos, entregando a previsão atualizada antes de o mercado abrir.

*\!important\!*

Não se trata de quatro modelos de Inteligência Artificial independentes conversando entre si, mas sim de **três blocos neurais** formando uma única arquitetura de ponta a ponta (*end-to-end*), orquestrados por **um pipeline de engenharia de software** (o motor de re-treino). 

### **1\. O Fluxo da Arquitetura Neural**

**Bloco 1: O Extrator de Visão Temporal (TensorFlow)** Recebe matrizes numéricas (séries contendo preço de abertura, máxima, mínima, fechamento e volume financeiro) divididas em janelas de tempo. Seu trabalho é puramente processar a inércia matemática. A saída deste bloco não é uma previsão, mas sim um vetor de características (um tensor unidimensional de números flutuantes) que encapsula de forma comprimida o "estado técnico e o momentum" da ação naquele momento.

**Bloco 2: O Extrator de Linguagem Causal (PyTorch)** Recebe strings de texto pré-processadas. Utiliza uma arquitetura baseada em Transformers focada em finanças (como um FinBERT) para extrair o sentimento, o peso e a causalidade do contexto macroeconômico e corporativo. Semelhante ao Bloco 1, sua saída é um segundo vetor de características numéricas, encapsulando o *"estado fundamentalista e de risco externo"*.

**Bloco 3: A Camada de Fusão e Inferência (O Tomador de Decisão)** Este é o ápice da rede, que pode ser construído como uma rede neural densa (Multilayer Perceptron) na ponta do PyTorch. Ele recebe os dois vetores gerados simultaneamente pelos Blocos 1 e 2 e os concatena em um único vetor massivo. Este vetor passa por camadas ocultas que aprendem a correlação entre o gráfico e a notícia. A camada final de saída contém os neurônios que projetam a distribuição de probabilidade das previsões para os horizontes de 7 dias, 3 meses e 1 ano.

**O Motor de Online Learning (O Pipeline de Atualização)** 

O re-treino de 15 dias não é uma IA separada, mas uma esteira de automação (um script de operações de aprendizado de máquina). Quando o sistema identifica a defasagem temporal, ele executa um gatilho algorítmico: extrai os dados recentes (OHLCV e textos), congela os pesos matemáticos dos Blocos 1 e 2 (para que a IA não sofra "esquecimento catastrófico" de tudo o que aprendeu em 10 anos) e aplica um micro-treinamento rápido e focado apenas nos pesos do Bloco 3, ajustando a calibração preditiva à volatilidade exata daquela quinzena. 

### **2\. A Engenharia de Extração de Notícias**

A captura de dados qualitativos exige precisão de infraestrutura. Injetar texto bruto e sujo no **PyTorch** destrói a acurácia de qualquer fundo quantitativo. O processo deve abandonar o conceito frágil de *web scraping* em portais de notícias (onde o HTML muda frequentemente) e adotar uma via estruturada.

* **Aquisição via APIs REST** A alimentação do modelo deve ser feita consumindo endpoints de APIs financeiras. Para o ambiente macroeconômico e internacional, agregações via serviços como Alpha Vantage, NewsAPI, ou as rotinas abertas da biblioteca `yfinance` entregam respostas em JSON contendo título, corpo do texto, fonte e o *timestamp* exato da publicação. Para o microambiente (as 7 empresas), o consumo de dados abertos da Comissão de Valores Mobiliários (CVM) e sistemas de Relações com Investidores (RI) automatiza a extração de Fatos Relevantes e emissões de debêntures.

* **Sanitização e Padronização Algorítmica** Antes de tocar na IA, o JSON passa por um script de limpeza. Removem-se formatações residuais, e aplica-se um controle de entidades nomeadas. O algoritmo garante que termos variados ("Banco Central", "Bacen", "Copom") sejam convertidos para um token único e coerente, e que as métricas financeiras dentro do texto sejam isoladas da prosa do jornalista.

* **Alinhamento Temporal Assíncrono (O Fator Crítico)** A parte mais complexa da engenharia de dados com o uso de bibliotecas de manipulação de dataframes (como Pandas) é atrelar a notícia à distorção correta do gráfico. O mercado financeiro opera com *gaps* de informação. Se um Fato Relevante da Petrobras é publicado às 20h00, o mercado à vista já está fechado. O pipeline de dados deve atrelar este texto obrigatoriamente à variação do *gap* de abertura das 10h00 da manhã do dia útil seguinte, e não ao *candle* de fechamento do dia da notícia. Um alinhamento errado destrói a percepção de causa e efeito do modelo.

* **Tokenização**: Finalmente, o texto alinhado e limpo é fragmentado pelo tokenizador nativo do modelo PyTorch, transformando palavras em matrizes de identificadores numéricos. É este bloco estruturado que atravessa a rede neural do Bloco 2 para virar o vetor de contexto político e causal.

## **O Eixo de Auditoria: O Conselheiro**

Com esses dados até aqui, temos um modelo ‘preditivo’ e 'especulador'; o que é muito bom para cumprir aquela missão de *"recomendar ou vetar a compra de uma ação em específico por um período em específico".* Suponhamos que a previsão da Petrobras para os próximos 3 meses é de 28% de valorização; faço essa compra e deixo as ações na carteira. Os 3 meses se passam e o valor útil daquela previsão se encerra. para concluir a transação no final daquele período estipulado, o user deve ser notificado que "seu ciclo de lucro nessa stock atingiu o fim ciclo; recomendamos a venda de X ações obtendo um lucro de R$XYZ em relação ao valor comprado no dia XX/YY/ZZZZ, em comparação com a data de hoje WW/HH/JJJJ computamos que o ideal é recolher o lucro dessa operação”.

Então, de forma independente e separada, nós temos uma espécie de 'consultor' que requer um outro treinamento específico para ser um 'consultor treinado em obter lucro com trading de curto prazo’. Você concorda com essa observação? como podemos criar essa implementação nesse sistema ? faz sentido ?

A arquitetura deste eixo abandona a previsão estocástica em favor da rigidez matemática focada na gestão de risco e no Valor Esperado (EV). Enquanto os modelos preditivos identificam a vantagem estatística para a entrada na operação, o Conselheiro atua de forma independente como um motor determinístico de saída.

A premissa de aguardar passivamente o fim de um ciclo estipulado (ex: 3 meses para 28% de lucro) configura uma ineficiência estratégica. Se o ativo precifica 24% de alta em apenas 20 dias, manter o capital exposto à volatilidade do mercado pelos 70 dias restantes para buscar 4% adicionais inverte a relação de risco, gerando um EV negativo. O lucro não realizado evapora se a execução hesitar.

Para solucionar isso, o sistema implementa um Agente de Execução assíncrono (Worker/Daemon) que audita a carteira em tempo real. Ele monitora simultaneamente o lucro não realizado (PnL), o tempo de ciclo restante e a volatilidade imediata. Utilizando um *Trailing Stop* dinâmico baseado em equações estritas de risco-retorno, o Conselheiro corta a operação no milissegundo em que a vantagem matemática desaparece.

A notificação gerada ao usuário deixa de ser um alerta passivo de fim de prazo e torna-se um relatório de execução tática: *"Ciclo de risco/retorno otimizado antecipadamente. Lucro de R$ XYZ (24%) garantido. A manutenção da posição passou a apresentar EV negativo face à volatilidade atual. Operação liquidada para proteção de caixa."* Isso garante que a aplicação não apenas indique o alvo, mas proteja o capital com frieza algorítmica.

## **A Camada Alpha: Validação Operacional e Uso Real Controlado**

Até este ponto, a arquitetura descreve como a IA pode aprender padrões de preço, interpretar contexto externo e sugerir decisões. Porém, para que o projeto deixe de ser apenas uma demonstração acadêmica e se torne uma ferramenta alpha utilizável no dia a dia, existe uma camada indispensável: a **Validação Operacional Financeira**.

Essa camada é a ponte entre "o modelo fez uma previsão" e "eu posso testar essa previsão com responsabilidade". Ela não aumenta a complexidade visual do projeto, mas aumenta muito a segurança e a credibilidade do sistema. Para o trabalho acadêmico, ela demonstra maturidade técnica. Para o uso pessoal, ela evita que uma previsão aparentemente convincente seja tratada como certeza.

O objetivo do alpha não é entregar a arquitetura completa de um fundo quantitativo, nem revelar toda a tese estratégica do produto. A limitação do MVP é nossa aliada: vamos construir apenas o necessário para provar o fluxo completo, medir resultados e permitir uso pessoal controlado, preservando a ideia principal e evitando excesso de exposição.

### **1. O Problema da Previsão Bonita, mas Financeiramente Frágil**

Um modelo pode acertar a direção de uma ação e ainda assim gerar prejuízo. Isso acontece porque o mundo real possui custos, atrasos e restrições que o gráfico histórico puro não mostra com fidelidade.

Os principais riscos são:

* **Custos operacionais:** taxas, emolumentos e impostos reduzem o retorno líquido.
* **Spread:** diferença entre preço de compra e venda.
* **Slippage:** diferença entre o preço esperado e o preço realmente executado.
* **Liquidez:** nem sempre é possível entrar ou sair no preço ideal.
* **Gaps de abertura:** uma notícia fora do pregão pode fazer o ativo abrir muito acima ou abaixo do fechamento anterior.
* **Overfitting:** o modelo pode parecer excelente no passado porque decorou padrões históricos, mas falhar no futuro.
* **Viés de sobrevivência e seleção:** escolher apenas ativos ou períodos em que a estratégia funcionou distorce a avaliação.

Portanto, a pergunta central do alpha não é apenas: *"o modelo prevê?"* A pergunta correta é: *"a previsão continua útil depois de custos, risco, atraso de execução e comparação com estratégias simples?"*

### **2. Backtest, Walk-Forward e Paper Trading**

O sistema deve passar por três filtros antes de qualquer teste com capital real, mesmo que pequeno.

**Backtest Histórico:** simula como o modelo teria se comportado no passado. Ele deve comparar o desempenho da estratégia contra referências simples, como comprar e segurar o próprio ativo (*buy and hold*) e o Ibovespa. O objetivo é evitar que a IA pareça sofisticada, mas perca para uma estratégia passiva.

**Walk-Forward Validation:** divide o tempo em janelas. O modelo treina em um período passado e é testado no período imediatamente seguinte, repetindo esse processo várias vezes. Isso é mais realista do que treinar em todo o histórico e testar aleatoriamente, pois respeita a ordem temporal do mercado.

**Paper Trading:** é o estágio mais importante para o uso pessoal. O sistema emite sinais reais no dia atual, registra tudo em banco de dados, mas não executa dinheiro real automaticamente. Ele acompanha o resultado como se a operação tivesse sido feita. Esse mecanismo mostra como o modelo se comporta fora do laboratório, com dados novos e sem olhar o futuro.

O alpha funcional deve nascer com paper trading. Assim, o usuário pode usar a ferramenta diariamente, comparar o sinal da IA com sua própria análise e decidir se quer ou não testar com capital simbólico posteriormente.

### **3. Política de Tamanho de Posição (Position Sizing)**

Uma lacuna crítica em sistemas de previsão é dizer *"qual ativo comprar"*, mas não dizer *"quanto comprar"*. No mercado real, o tamanho da posição é tão importante quanto a direção prevista.

O MVP deve conter uma política simples:

* Definir capital total da carteira de teste.
* Definir risco máximo por operação, por exemplo 0,5% ou 1% do capital.
* Calcular a distância entre preço de entrada e stop loss.
* Determinar a quantidade máxima que pode ser comprada sem ultrapassar o risco permitido.

Exemplo: se a carteira alpha possui R$ 10.000 e o risco máximo por operação é 1%, a perda máxima aceitável é R$ 100. Se a distância entre entrada e stop é 5%, o valor máximo alocado naquela operação será R$ 2.000. Assim, mesmo que o stop seja atingido, a perda respeita o limite planejado.

Essa camada impede que o usuário coloque capital demais em uma previsão de baixa confiança.

### **4. O Modo "Não Operar"**

Um sistema financeiro maduro não deve ser obrigado a recomendar compra ou venda todos os dias. Em muitos momentos, a melhor decisão é não fazer nada.

Portanto, além de "comprar", "vender", "manter" ou "sair", a IA deve possuir uma saída explícita: **não operar**.

O modo "não operar" deve ser acionado quando:

* a confiança do modelo estiver abaixo do mínimo;
* houver dados faltantes ou desatualizados;
* o risco/retorno projetado for ruim;
* o ativo estiver em volatilidade extrema;
* os modelos técnico e qualitativo divergirem fortemente;
* existir evento crítico próximo, como balanço, Copom ou decisão política relevante;
* o stop necessário estiver distante demais, tornando a posição cara ou arriscada.

Essa função é fundamental para o uso real, porque protege o usuário de operar por ansiedade. O valor do sistema não está apenas nas operações que ele sugere, mas também nas operações que ele veta.

### **5. Registro Imutável de Teses**

Cada previsão deve ser salva como uma tese imutável. Isso cria auditoria e impede que a análise seja reescrita depois do resultado.

Uma tese deve conter:

* data da previsão;
* ativo;
* horizonte;
* direção esperada;
* retorno esperado;
* confiança;
* preço de referência;
* entrada sugerida;
* stop loss;
* alvo parcial;
* alvo principal;
* tamanho máximo de posição;
* justificativa técnica;
* justificativa contextual;
* versão do modelo;
* dados usados até determinada data;
* resultado observado após o prazo.

Esse histórico transforma o app em uma ferramenta de aprendizado. O usuário passa a enxergar onde a IA acerta, onde erra, em quais ativos funciona melhor e em quais situações deve ser ignorada.

### **6. Causa, Correlação e Linguagem Responsável**

O bloco PyTorch pode associar notícias a movimentos de preço, mas isso não significa que ele prove causalidade absoluta. Em finanças, múltiplos fatores agem ao mesmo tempo.

Por isso, a interface deve evitar frases como:

* "a causa da queda foi..."
* "o ativo caiu porque..."
* "o modelo identificou o motivo real..."

E deve preferir frases como:

* "evento associado ao movimento";
* "contexto compatível com a variação";
* "hipótese causal provável";
* "fator externo relevante no período".

Essa escolha de linguagem torna o produto mais honesto e tecnicamente defensável.

### **7. Métricas Mínimas para o Alpha**

O alpha funcional deve exibir métricas simples, mas suficientes para avaliação diária:

* acurácia direcional;
* retorno acumulado;
* retorno líquido estimado após custos;
* drawdown máximo;
* taxa de operações vencedoras;
* ganho médio;
* perda média;
* relação risco/retorno;
* comparação contra buy and hold;
* quantidade de sinais bloqueados por "não operar".

Essas métricas devem orientar a evolução do projeto. Se a IA não supera um baseline simples, ela ainda pode ser útil como ferramenta de estudo, mas não deve ser tratada como motor de decisão.

### **8. Delimitação Estratégica do MVP para Apresentação**

Como o objetivo também é apresentar um trabalho sobre PyTorch e TensorFlow, o MVP deve ser propositalmente enxuto. Ele deve demonstrar a integração dos frameworks sem expor todos os detalhes de uma possível implementação comercial.

O recorte recomendado para apresentação é:

* TensorFlow analisando séries históricas de preço;
* PyTorch classificando sentimento ou contexto textual em amostra controlada;
* fusão simplificada entre sinais técnico e qualitativo;
* dashboard com previsão probabilística;
* paper trading e política de risco como camada de segurança;
* aviso claro de que se trata de alpha experimental.

O que fica fora do MVP:

* automação real de ordens;
* integração direta com corretora;
* arquitetura completa de deploy em escala;
* modelos proprietários avançados;
* estratégias detalhadas de otimização de carteira.

Essa limitação protege a ideia e reduz a complexidade. O alpha precisa provar que o fluxo funciona, não entregar todas as respostas finais.

### **Conclusão da Camada Alpha**

O verdadeiro produto não é apenas uma IA que tenta prever ações. O verdadeiro produto é um sistema experimental de decisão financeira que combina:

* previsão quantitativa;
* interpretação textual;
* validação histórica;
* simulação em tempo real;
* controle de risco;
* auditoria de teses;
* capacidade de dizer "não operar".

Essa camada torna o projeto mais realista, mais seguro e mais convincente para apresentação. Ela também permite que o usuário teste a ferramenta no dia a dia sem transformar uma hipótese acadêmica em exposição financeira irresponsável.

