# Sprint Plan - Profit App

## 1. Leitura do Material de Apoio

O material define uma aplicacao de inteligencia artificial para apoio a decisoes de investimento nas principais empresas da B3. A ideia central e usar PyTorch e TensorFlow de forma complementar:

- TensorFlow para o eixo quantitativo, lendo series temporais de preco e volume.
- PyTorch para o eixo qualitativo, lendo noticias, fatos relevantes e contexto macroeconomico.
- Um modulo de fusao para combinar os dois vetores de informacao e gerar previsoes.
- Um motor de re-treino incremental para atualizar o sistema quando o usuario fica dias sem usar a aplicacao.
- Um conselheiro de risco para acompanhar operacoes abertas e recomendar saidas quando o valor esperado deixa de compensar o risco.

O ponto mais importante: o projeto deve nascer como uma plataforma operacional, auditavel e mensuravel, capaz de testar hipoteses com dados historicos, explicar suas previsoes e acompanhar resultados em producao com trilha completa de risco.

Para este trabalho, a restricao de escopo e uma vantagem. O objetivo nao e expor uma arquitetura comercial completa, nem construir algo excessivamente complexo. O objetivo e criar um alpha funcional: pequeno, utilizavel no dia a dia pelo proprio autor, bom o suficiente para testar sinais em paper trading e claro o bastante para apresentar o papel de PyTorch e TensorFlow.

## 2. Objetivo do Produto

Construir uma aplicacao funcional que acompanhe um grupo inicial de acoes brasileiras, gere previsoes para 7 dias, 3 meses e 1 ano, relacione movimentos de preco com eventos externos e apresente recomendacoes de entrada, saida, risco e acompanhamento de posicoes.

O app deve responder quatro perguntas principais:

1. O ativo tende a subir, cair ou ficar lateralizado em cada horizonte?
2. Qual e a confianca estatistica dessa previsao?
3. Quais fatores tecnicos e externos sustentam essa leitura?
4. Se uma operacao for aberta, quando faz sentido sair antes do prazo original?

## 3. Escopo Inicial

O escopo inicial deve ser menor que a visao completa, para permitir entrega real.

### Ativos iniciais

Comecar com 7 acoes liquidas da B3. A lista pode ser ajustada conforme disponibilidade de dados, mas uma base inicial plausivel e:

- PETR4 - Petrobras
- VALE3 - Vale
- ITUB4 - Itau Unibanco
- BBDC4 - Bradesco
- BBAS3 - Banco do Brasil
- ABEV3 - Ambev
- WEGE3 - WEG

### Horizontes de previsao

- 7 dias: curto prazo, sensivel a momentum, volume, volatilidade e sentimento recente.
- 3 meses: medio prazo, sensivel a balancos, juros, commodities e revisoes de expectativa.
- 1 ano: longo prazo, sensivel a fundamentos, ciclos macroeconomicos e tendencias setoriais.

### Tipo de saida

Evitar saidas absolutas como "compre" ou "vai subir 15%". A saida deve ser probabilistica:

- Direcao esperada: alta, queda ou lateralidade.
- Retorno esperado em percentual.
- Intervalo de confianca.
- Risco estimado.
- Entrada sugerida.
- Stop loss sugerido.
- Alvo parcial e alvo principal.
- Justificativa tecnica e contextual.

## 4. Arquitetura Conceitual

### 4.1 Bloco Quantitativo - TensorFlow

Responsabilidade: estudar o comportamento historico do preco.

Entrada:

- OHLCV: abertura, maxima, minima, fechamento e volume.
- Indicadores derivados: retorno diario, medias moveis, volatilidade, RSI, drawdown, volume relativo.
- Janelas temporais: por exemplo 30, 60, 120 e 252 pregoes.

Modelo inicial recomendado:

- MVP: modelo tabular ou sequencial simples, como MLP ou GRU pequena.
- Evolucao: TCN ou Time-Series Transformer.

Saida:

- Embedding tecnico do ativo.
- Probabilidade tecnica de alta, queda ou lateralidade.
- Sinais de anomalia: volume incomum, rompimento, volatilidade extrema, gap.

### 4.2 Bloco Qualitativo - PyTorch

Responsabilidade: estudar noticias, fatos relevantes e contexto externo.

Entrada:

- Noticias financeiras com timestamp.
- Comunicados da CVM e relacoes com investidores.
- Eventos macroeconomicos: Selic, Copom, cambio, Brent, minerio, juros americanos.

Modelo inicial recomendado:

- MVP: classificacao de sentimento com modelo pre-treinado ou heuristica simples.
- Evolucao: FinBERT ou transformer financeiro em PyTorch via Hugging Face.

Saida:

- Embedding de sentimento e contexto.
- Polaridade: positiva, negativa ou neutra.
- Relevancia do evento para cada ativo.
- Possivel causa associada a variacoes relevantes no grafico.

### 4.3 Bloco de Fusao e Inferencia

Responsabilidade: combinar o estado tecnico com o estado contextual.

Entrada:

- Embedding tecnico do TensorFlow.
- Embedding textual do PyTorch.
- Features auxiliares de calendario, setor e macroeconomia.

Modelo inicial recomendado:

- MLP em PyTorch ou TensorFlow.
- Para simplificar o MVP, a fusao pode comecar como uma camada densa que recebe features numericas ja calculadas.

Saida:

- Previsoes para 7 dias, 3 meses e 1 ano.
- Retorno esperado.
- Intervalo de risco.
- Score de confianca.
- Explicacao resumida.

### 4.4 Conselheiro de Risco

Responsabilidade: acompanhar operacoes abertas e recomendar manutencao, realizacao parcial, venda total ou ajuste de stop.

Esse modulo nao precisa ser uma IA profunda no inicio. Para o MVP, ele deve ser deterministico e baseado em regras de risco.

Entradas:

- Preco de entrada.
- Quantidade comprada.
- Horizonte da tese.
- Alvo previsto.
- Stop loss inicial.
- Preco atual.
- Volatilidade recente.
- Tempo restante da tese.
- Lucro ou prejuizo nao realizado.

Regras iniciais:

- Se o ativo atingir grande parte do alvo antes do prazo, sugerir realizacao parcial ou total.
- Se a volatilidade aumentar e o retorno restante for pequeno, reduzir exposicao.
- Se o preco romper o stop, recomendar saida.
- Se o modelo atualizar a previsao contra a posicao aberta, alertar o usuario.

Saida:

- Manter posicao.
- Realizar parcial.
- Encerrar posicao.
- Ajustar stop.
- Reavaliar tese.

### 4.5 Motor de Re-treino Incremental

Responsabilidade: atualizar a calibragem do sistema sem treinar tudo do zero.

Fluxo:

1. Detectar ultima data de atualizacao.
2. Baixar dados novos desde essa data.
3. Atualizar base local.
4. Recalcular features.
5. Congelar extratores principais.
6. Ajustar apenas as camadas finais ou calibradores.
7. Registrar metricas antes e depois do ajuste.

Para o MVP, o re-treino pode ser simplificado como reprocessamento dos dados recentes e recalibragem estatistica. O fine-tuning neural entra depois que a base historica e as metricas estiverem confiaveis.

### 4.6 Camada de Validacao Operacional Alpha

Responsabilidade: impedir que uma previsao tecnicamente interessante seja usada como decisao financeira sem controle de risco.

Essa camada e obrigatoria para o alpha, porque o objetivo e permitir uso pessoal controlado no dia a dia. Ela nao precisa ser complexa, mas precisa ser rigida.

Componentes:

- Backtest historico.
- Walk-forward validation.
- Paper trading.
- Simulador de custos, spread e slippage.
- Politica de tamanho de posicao.
- Modo "nao operar".
- Registro imutavel de teses.
- Comparacao contra buy and hold e Ibovespa.

Saidas:

- Sinal aprovado para simulacao.
- Sinal bloqueado por risco.
- Sinal bloqueado por dados incompletos.
- Sinal classificado como "nao operar".
- Relatorio de desempenho do alpha.

Regras iniciais:

- Nenhuma previsao pode virar operacao simulada sem stop, alvo, confianca e tamanho maximo de posicao.
- Nenhuma previsao deve virar acao operacional se perder para baselines simples.
- Toda tese deve ser salva antes do resultado acontecer.
- O sistema deve conseguir dizer "nao operar" quando a relacao risco/retorno for ruim.
- O alpha deve registrar decisao, tese, sizing, risco e estado operacional antes de qualquer acao de carteira.

## 5. Arquitetura Tecnica Proposta

### Backend

- Python.
- FastAPI para expor endpoints.
- Pandas e NumPy para processamento.
- yfinance ou fonte equivalente para OHLCV.
- TensorFlow/Keras para o bloco quantitativo.
- PyTorch/Hugging Face para o bloco qualitativo.
- PostgreSQL como banco operacional.
- APScheduler, Celery ou RQ para jobs de atualizacao e auditoria.

### Frontend

- Streamlit para MVP rapido ou React/Next.js para produto mais robusto.
- Dashboard com ativos, previsoes, carteira simulada, alertas e historico.

### Armazenamento

- Tabelas de ativos.
- Historico OHLCV.
- Noticias e eventos.
- Features calculadas.
- Previsoes geradas.
- Operacoes simuladas.
- Logs de re-treino.
- Metricas de backtest.

### Observacao importante

Como o projeto envolve investimento, o app deve exibir previsoes com tese, risco, custos, confianca, decisao operacional e justificativa auditavel.

Para preservar a ideia e manter a entrega viavel, o trabalho deve apresentar um MVP controlado. A demonstracao deve mostrar o fluxo essencial com PyTorch e TensorFlow, mantendo estrategias proprietarias em camada interna e priorizando rastreabilidade financeira.

## 6. Backlog de Sprints

Cada sprint foi pensada para entregar uma parte funcional e testavel. A ordem evita comecar pela IA mais complexa antes de ter dados, metricas e uma interface minima.

## Sprint 0 - Fundacao do Projeto

Objetivo: transformar a ideia em um repositorio executavel.

Tarefas:

- Criar estrutura inicial do projeto.
- Definir ambiente Python.
- Criar arquivo de dependencias.
- Criar configuracao basica de lint/testes.
- Definir arquitetura de pastas.
- Criar README com objetivo, aviso de risco e como executar.

Entregaveis:

- Projeto executando localmente.
- Ambiente reproduzivel.
- Estrutura inicial documentada.

Criterio de pronto:

- Um novo desenvolvedor consegue instalar dependencias e rodar um comando basico do projeto.

## Sprint 1 - Coleta de Dados de Mercado

Objetivo: obter historico confiavel dos ativos.

Tarefas:

- Definir os 7 tickers iniciais.
- Criar coletor de OHLCV.
- Salvar dados em CSV ou SQLite.
- Normalizar datas, timezone e calendario de pregao.
- Criar rotina de atualizacao incremental.
- Validar dados faltantes e duplicados.

Entregaveis:

- Base historica dos 7 ativos.
- Script de atualizacao.
- Relatorio simples de qualidade dos dados.

Criterio de pronto:

- O sistema baixa, salva e atualiza dados de preco sem duplicar registros.

## Sprint 2 - Feature Engineering Quantitativo

Objetivo: transformar preco bruto em sinais tecnicos.

Tarefas:

- Calcular retornos diarios.
- Calcular medias moveis.
- Calcular volatilidade historica.
- Calcular volume relativo.
- Calcular drawdown.
- Criar labels para 7 dias, 3 meses e 1 ano.
- Separar treino, validacao e teste respeitando ordem temporal.

Entregaveis:

- Dataset numerico pronto para treino.
- Funcoes reutilizaveis de features.
- Primeiras metricas estatisticas por ativo.

Criterio de pronto:

- Cada linha do dataset possui features atuais e alvos futuros sem vazamento de informacao.

## Sprint 3 - Modelo Quantitativo TensorFlow MVP

Objetivo: criar primeira previsao baseada apenas em preco.

Tarefas:

- Criar baseline simples.
- Treinar modelo TensorFlow inicial.
- Gerar previsoes para os tres horizontes.
- Avaliar direcao e erro percentual.
- Salvar modelo treinado.
- Criar script de inferencia.

Entregaveis:

- Modelo TensorFlow treinado.
- Metricas de validacao.
- Endpoint ou funcao de previsao quantitativa.

Criterio de pronto:

- O sistema recebe um ticker e retorna previsao tecnica para 7 dias, 3 meses e 1 ano.

## Sprint 4 - Backtesting Basico

Objetivo: medir se a previsao teria funcionado no passado.

Tarefas:

- Criar simulador historico de entradas e saidas.
- Comparar modelo contra buy and hold.
- Calcular retorno acumulado.
- Calcular drawdown maximo.
- Calcular taxa de acerto direcional.
- Calcular Sharpe simplificado.

Entregaveis:

- Relatorio de backtest por ativo.
- Comparacao com estrategias basicas.
- Identificacao de limites do modelo.

Criterio de pronto:

- Toda previsao exibida no app pode ser comparada com metricas historicas do proprio modelo.

## Sprint 5 - Validacao Financeira e Paper Trading Alpha

Objetivo: transformar previsoes em sinais testaveis com seguranca operacional.

Tarefas:

- Criar motor de paper trading.
- Registrar sinais diarios de forma imutavel.
- Simular custo operacional, spread e slippage.
- Implementar calculo de tamanho maximo de posicao.
- Definir risco maximo por operacao.
- Criar regra de "nao operar".
- Comparar sinais contra buy and hold e Ibovespa.
- Criar relatorio semanal de desempenho.
- Bloquear operacoes simuladas sem stop, alvo e confianca minima.

Entregaveis:

- Carteira simulada alpha.
- Registro historico de teses.
- Relatorio de performance fora da amostra.
- Politica simples de risco por operacao.

Criterio de pronto:

- O sistema consegue emitir um sinal real no dia atual, salvar a tese, simular entrada, calcular posicao maxima e acompanhar o resultado sem usar dinheiro real.

## Sprint 6 - Coleta de Noticias e Eventos

Objetivo: montar a base qualitativa.

Tarefas:

- Escolher fonte inicial de noticias ou eventos.
- Criar coletor via API sempre que possivel.
- Salvar titulo, corpo, fonte, ticker relacionado e timestamp.
- Criar limpeza basica de texto.
- Criar normalizacao de entidades como Copom, Bacen e Banco Central.
- Criar regra de alinhamento temporal com o proximo pregao quando a noticia sair fora do horario de mercado.

Entregaveis:

- Base inicial de noticias/eventos.
- Pipeline de limpeza.
- Funcao de alinhamento noticia-pregao.

Criterio de pronto:

- Uma noticia publicada fora do horario de mercado e associada ao pregao correto seguinte.

## Sprint 7 - Modelo Qualitativo PyTorch MVP

Objetivo: extrair sentimento e contexto dos textos.

Tarefas:

- Integrar modelo pre-treinado de NLP.
- Tokenizar textos.
- Gerar score de sentimento.
- Gerar embedding textual.
- Associar sentimento agregado por ativo e data.
- Criar avaliacao manual de amostra pequena para verificar coerencia.

Entregaveis:

- Modelo PyTorch ou pipeline Hugging Face funcional.
- Scores de sentimento por ativo e data.
- Embeddings textuais persistidos.

Criterio de pronto:

- O sistema recebe textos de um ativo e retorna sentimento agregado com timestamp alinhado.

## Sprint 8 - Fusao dos Modelos

Objetivo: combinar informacao tecnica e qualitativa.

Tarefas:

- Unir dataset quantitativo com features qualitativas.
- Criar modelo de fusao inicial.
- Treinar previsoes conjuntas.
- Comparar performance contra o modelo apenas tecnico.
- Expor justificativa com principais sinais usados.

Entregaveis:

- Modelo de fusao treinado.
- Comparativo tecnico versus tecnico + noticias.
- Saida unificada por ativo e horizonte.

Criterio de pronto:

- O app consegue mostrar uma previsao que usa preco e contexto externo de forma combinada.

## Sprint 9 - API da Aplicacao

Objetivo: transformar modelos e dados em servico consumivel.

Tarefas:

- Criar API com FastAPI.
- Endpoint de lista de ativos.
- Endpoint de historico de preco.
- Endpoint de previsao por ticker.
- Endpoint de explicacao da previsao.
- Endpoint de atualizacao/re-treino.
- Endpoint de carteira simulada.
- Endpoint de registro de tese.
- Endpoint de sinais bloqueados por risco.
- Endpoint de metricas do paper trading.

Entregaveis:

- Backend executavel localmente.
- Documentacao Swagger automatica.
- Contratos de entrada e saida definidos.

Criterio de pronto:

- O frontend ou cliente HTTP consegue consumir previsoes e historico pela API.

## Sprint 10 - Interface MVP

Objetivo: permitir que um usuario use o sistema sem terminal.

Tarefas:

- Criar tela de dashboard.
- Mostrar cards ou tabela dos 7 ativos.
- Mostrar grafico historico.
- Mostrar previsoes por horizonte.
- Mostrar confianca, risco, entrada, alvo e stop.
- Criar area de justificativa tecnica/contextual.
- Criar modo de carteira simulada.
- Mostrar quando o sistema decidir "nao operar".
- Mostrar tamanho maximo de posicao sugerido.
- Mostrar retorno liquido estimado apos custos simulados.

Entregaveis:

- Interface funcional.
- Fluxo de consulta por ativo.
- Visualizacao basica de previsao e risco.

Criterio de pronto:

- O usuario seleciona PETR4, ve historico, previsao, justificativa, risco, tamanho maximo de posicao e pode simular uma entrada quando o sinal nao estiver bloqueado.

## Sprint 11 - Conselheiro de Risco

Objetivo: acompanhar posicoes simuladas e recomendar saida.

Tarefas:

- Criar modelo de posicao simulada.
- Registrar entrada, quantidade, preco, data, horizonte e tese.
- Atualizar preco atual periodicamente.
- Calcular PnL realizado e nao realizado.
- Implementar trailing stop.
- Implementar regras de valor esperado simplificado.
- Gerar notificacoes de manter, reduzir ou sair.
- Bloquear aumento de posicao quando o risco total da carteira estiver acima do limite.
- Sugerir "nao operar" quando a vantagem estatistica desaparecer.

Entregaveis:

- Worker de auditoria de carteira.
- Regras deterministicas de saida.
- Historico de alertas.

Criterio de pronto:

- Uma operacao simulada pode ser acompanhada e receber recomendacao de saida antes do fim do prazo.

## Sprint 12 - Re-treino e Atualizacao Incremental

Objetivo: atualizar o sistema apos defasagem de uso.

Tarefas:

- Registrar data da ultima atualizacao.
- Detectar defasagem superior a limite definido.
- Baixar dados novos.
- Reprocessar features.
- Recalibrar modelo final ou calibrador estatistico.
- Registrar metricas antes e depois.
- Exibir no app quando o modelo foi atualizado.

Entregaveis:

- Botao ou job de atualizacao.
- Log de re-treino.
- Indicador de frescor dos dados.

Criterio de pronto:

- Se o usuario ficar 15 dias sem usar, o sistema atualiza dados e recalibra previsoes antes de exibir nova recomendacao.

## Sprint 13 - Auditoria, Explicabilidade e Seguranca

Objetivo: tornar o sistema confiavel e auditavel.

Tarefas:

- Salvar cada previsao gerada.
- Salvar versao do modelo usado.
- Salvar dados de entrada principais.
- Criar historico de acertos e erros.
- Exibir painel de risco financeiro.
- Criar limites de confianca minima.
- Bloquear recomendacoes quando dados estiverem incompletos.
- Salvar custos simulados usados em cada operacao.
- Salvar decisao de "operar" ou "nao operar" com motivo.
- Evitar linguagem de causalidade absoluta em noticias.

Entregaveis:

- Trilha de auditoria.
- Tela de historico de previsoes.
- Regras de seguranca para recomendacoes fracas.

Criterio de pronto:

- Qualquer previsao antiga pode ser revisitada com dados, versao do modelo, custos simulados, decisao tomada e resultado observado.

## Sprint 14 - Produto Alpha Apresentavel

Objetivo: consolidar a aplicacao como alpha funcional, utilizavel pelo autor em rotina pessoal de testes e apresentavel como trabalho sobre PyTorch e TensorFlow.

Tarefas:

- Melhorar layout.
- Tratar erros da API.
- Criar testes automatizados principais.
- Criar seed/demo local.
- Documentar limitacoes.
- Criar roteiro de demonstracao.
- Criar status claro de execucao, carteira e risco operacional.
- Criar modo demo com dados suficientes para apresentacao.
- Separar o que sera apresentado do que ficara como evolucao futura.

Entregaveis:

- App alpha navegavel.
- Documentacao de uso.
- Demo com fluxo completo.
- Relatorio de limites e proximos passos.

Criterio de pronto:

- Um avaliador consegue rodar o app, consultar ativos, ver previsoes, entender o papel de TensorFlow e PyTorch, simular entrada, ver bloqueios de risco e acompanhar recomendacoes de saida.

## 7. Ordem Recomendada de Implementacao

Para nao travar no excesso de ambicao, a ordem pratica deve ser:

1. Dados de preco.
2. Features tecnicas.
3. Modelo TensorFlow simples.
4. Backtest.
5. Paper trading e validacao financeira alpha.
6. Interface basica.
7. Noticias e eventos.
8. Modelo PyTorch de sentimento.
9. Fusao dos modelos.
10. Carteira simulada.
11. Conselheiro de risco.
12. Re-treino incremental.
13. Auditoria e alpha apresentavel.

## 8. MVP Realista

O primeiro MVP nao precisa ter tudo. Ele deve conter:

- Coleta de OHLCV dos 7 ativos.
- Modelo TensorFlow simples para previsao direcional.
- Backtest basico.
- Paper trading com registro imutavel de teses.
- Politica de risco e tamanho maximo de posicao.
- Modo "nao operar".
- Dashboard com previsao por ativo.
- Carteira simulada.
- Regras simples de stop e alvo.

O PyTorch entra logo depois, quando ja existir uma base quantitativa funcionando. Isso reduz risco tecnico e permite provar primeiro que o sistema consegue baixar dados, treinar, prever, medir e exibir resultados.

Para a apresentacao academica, o PyTorch pode entrar em uma versao controlada: classificacao de sentimento em amostras de noticias ou eventos ja coletados. Isso demonstra o papel do framework sem exigir uma infraestrutura completa de NLP financeiro em producao.

## 9. Riscos Tecnicos

- Dados financeiros podem vir incompletos ou inconsistentes.
- Noticias com timestamp errado podem destruir a relacao causal.
- Modelos podem sofrer overfitting em poucos ativos.
- Previsao de 1 ano pode ser muito instavel para uma base pequena.
- Misturar PyTorch e TensorFlow aumenta complexidade de ambiente.
- O app pode parecer preciso mesmo quando nao tem significancia estatistica.
- Um backtest sem custos pode superestimar retorno.
- Um sinal sem sizing pode induzir exposicao excessiva.
- O usuario pode operar em momentos em que o sistema deveria ficar em silencio.

Mitigacoes:

- Comecar com modo simulado.
- Medir tudo por backtest.
- Comparar contra baselines simples.
- Salvar logs de previsao.
- Exibir intervalos de confianca.
- Bloquear recomendacoes com baixa confianca.
- Simular custos, spread e slippage.
- Usar validacao operacional antes de ampliar capital.
- Calcular tamanho maximo de posicao.
- Implementar decisao explicita de "nao operar".

## 10. Definicao de Sucesso

O projeto sera considerado funcional quando conseguir executar o fluxo completo:

1. Baixar dados atualizados dos ativos.
2. Processar indicadores e eventos.
3. Gerar previsoes por horizonte.
4. Exibir justificativa tecnica e contextual.
5. Validar o sinal contra regras de risco.
6. Calcular tamanho maximo de posicao.
7. Permitir simulacao de operacao em paper trading.
8. Acompanhar posicao aberta.
9. Recomendar saida, manutencao ou "nao operar" com base em risco.
10. Atualizar dados e recalibrar previsoes apos periodo de inatividade.
11. Registrar previsoes antigas para auditoria de acerto e erro.

## 11. Decisao Arquitetural Inicial

A decisao mais segura para o ponto zero e nao construir imediatamente a arquitetura final completa. O caminho correto e criar um nucleo funcional pequeno, medido e expansivel.

Primeira versao recomendada:

- Backend Python com FastAPI.
- Banco SQLite.
- Coleta OHLCV com yfinance ou fonte equivalente.
- TensorFlow para modelo tecnico inicial.
- Dashboard em Streamlit para acelerar a entrega.
- Carteira simulada com regras deterministicas.
- Paper trading local.
- Registro imutavel de teses.
- Politica simples de risco por operacao.
- Nenhuma integracao com corretora.

Segunda versao:

- Adicionar PyTorch/Hugging Face para sentimento.
- Criar fusao tecnica + textual.
- Melhorar explicabilidade.
- Melhorar validacao walk-forward.
- Trocar ou evoluir frontend se necessario.

Terceira versao:

- Re-treino incremental.
- Worker assicrono.
- Banco PostgreSQL.
- Monitoramento de modelos.
- Deploy.

O alpha apresentavel deve parar antes da automacao real de ordens. Essa fronteira protege o projeto, reduz risco juridico e mantem a apresentacao focada no uso combinado de PyTorch e TensorFlow.

## 12. Proxima Acao Recomendada

A proxima acao concreta e iniciar a Sprint 0 criando a estrutura do projeto. Uma estrutura simples e suficiente para comecar:

```text
profit-app/
  app/
    api/
    data/
    features/
    models/
    services/
    workers/
  notebooks/
  tests/
  storage/
  README.md
  requirements.txt
```

Depois disso, a Sprint 1 deve implementar o coletor de dados dos 7 ativos. Sem dados confiaveis, nenhum modelo posterior tera valor.