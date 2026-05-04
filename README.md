# Profit App

Alpha experimental para avaliacao e predicao de investimentos na B3 usando dados reais, paper trading, eventos/noticias e controle de risco.

## Aviso

Este projeto e educacional e experimental. As previsoes geradas nao constituem recomendacao financeira profissional. A primeira versao opera apenas com coleta de dados e simulacao.

## Sprint atual

Sprints 1 a 11: coleta de dados OHLCV, geracao de features tecnicas, treino TensorFlow, backtest basico, walk-forward validation, paper trading alpha, noticias/eventos, sentimento PyTorch MVP, fusao tecnico-contextual, API FastAPI, dashboard MVP e conselheiro de risco para carteira simulada.

## Ambientes Python

O projeto possui dois ambientes locais:

- `.venv`: ambiente leve usado inicialmente para coleta de dados.
- `.venv311`: ambiente Python 3.11 usado para TensorFlow, treino e backtest.

Use `.venv311` para qualquer comando de ML, inferencia, backtest, fusao e geracao de sinais operacionais.

Para rodar o pipeline operacional completo em uma chamada:

```powershell
.\scripts\run_ml_pipeline.ps1
```

Para reaproveitar modelos ja treinados e apenas recalcular inferencias/sinais:

```powershell
.\scripts\run_ml_pipeline.ps1 -SkipTraining
```

## Ativos iniciais

- PETR4.SA
- VALE3.SA
- ITUB4.SA
- BBDC4.SA
- BBAS3.SA
- ABEV3.SA
- WEGE3.SA

## Como atualizar os dados

```powershell
.\.venv\Scripts\python.exe -m app.cli update-prices
```

Para uma coleta curta de validacao:

```powershell
.\.venv\Scripts\python.exe -m app.cli update-prices --period 1mo
```

Para conferir quantas linhas foram salvas por ativo:

```powershell
.\.venv\Scripts\python.exe -m app.cli price-summary
```

Por padrao, os dados sao salvos em `storage/profit_app.sqlite3`.

## Como gerar features tecnicas

```powershell
.\.venv311\Scripts\python.exe -m app.cli generate-features
```

Para conferir o dataset tecnico por ativo e split temporal:

```powershell
.\.venv311\Scripts\python.exe -m app.cli feature-summary
```

## Teste E2E

```powershell
.\.venv311\Scripts\python.exe tests\e2e_alpha_pipeline.py
```

## Treino TensorFlow

O modelo atual usa features tecnicas relativas, one-hot por ticker, pesos de classe, batch normalization e dropout. A metrica de acuracia isolada nao e usada como decisao operacional; o criterio principal para liberar paper trading e o gate walk-forward.
O treino exibe as epocas no terminal e grava `epochs_ran` no metadata do modelo.

```powershell
.\.venv311\Scripts\python.exe -m app.cli train-tf-direction --epochs 40 --batch-size 64
```

Para listar modelos treinados:

```powershell
.\.venv311\Scripts\python.exe -m app.cli model-summary
```

## Backtest Basico

```powershell
.\.venv311\Scripts\python.exe -m app.cli run-backtest --threshold 0.45 --holding-days 7 --cost-per-trade 0.002
```

Para listar backtests:

```powershell
.\.venv311\Scripts\python.exe -m app.cli backtest-summary
```

Para escolher o threshold no conjunto de validacao e testar fora da amostra com filtro de drawdown:

```powershell
.\.venv311\Scripts\python.exe -m app.cli run-optimized-backtest
```

Esse modo e mais conservador que o threshold fixo. Ele pode decidir nao operar quando a validacao nao apresentar vantagem positiva com drawdown aceitavel.

Para executar a validacao walk-forward com recalibragem por janelas temporais e diagnostico por ticker:

```powershell
.\.venv311\Scripts\python.exe -m app.cli run-walk-forward
```

O paper trading usa essa validacao como trava operacional. Se o walk-forward reprovar retorno, comparacao contra buy and hold, estabilidade por janela ou dispersao por ativo, todos os sinais viram `no_operate`. Mesmo quando o walk-forward passa, cada sinal atual ainda precisa superar o threshold de probabilidade escolhido na validacao, confianca minima, retorno esperado liquido positivo e reward/risk minimo.

## Paper Trading Alpha

Gera teses imutaveis de paper trading a partir do ultimo modelo treinado. O comando calcula decisao `simulate_long` ou `no_operate`, stop, alvo, custo estimado, risco maximo e tamanho maximo de posicao. Ele nao executa ordens reais. O status esperado pode ser `no_operate` mesmo com o gate historico aprovado, quando o sinal do dia nao tem qualidade suficiente.

```powershell
.\.venv311\Scripts\python.exe -m app.cli generate-paper-signals
```

Para listar as teses salvas:

```powershell
.\.venv311\Scripts\python.exe -m app.cli paper-summary
```

## Noticias e Eventos

Cria eventos de exemplo para validar limpeza textual, normalizacao de entidades e alinhamento de noticia publicada apos o fechamento ao proximo pregao.

```powershell
.\.venv\Scripts\python.exe -m app.cli seed-sample-news
```

Para listar eventos salvos:

```powershell
.\.venv\Scripts\python.exe -m app.cli news-summary
```

## Sentimento PyTorch MVP

Gera features qualitativas por ativo e data alinhada. Em ambientes com PyTorch instalado, o pipeline usa tensores PyTorch para o embedding textual; no `.venv` Python 3.14 local, onde PyTorch ainda nao esta disponivel, ele usa fallback deterministico para manter o MVP executavel.

```powershell
.\.venv\Scripts\python.exe -m app.cli analyze-news-sentiment
.\.venv\Scripts\python.exe -m app.cli sentiment-summary
```

## Fusao Tecnico-Contextual

Combina a probabilidade tecnica do modelo TensorFlow com o sentimento qualitativo mais recente disponivel ate a data do sinal. A saida inclui direcao fusionada, score fusionado e JSON de explicacao.

```powershell
.\.venv311\Scripts\python.exe -m app.cli run-fusion
.\.venv311\Scripts\python.exe -m app.cli fusion-summary
```

## API FastAPI

Executa o backend local com dashboard em `http://127.0.0.1:8000/` e Swagger automatico em `http://127.0.0.1:8000/docs`.

```powershell
.\.venv311\Scripts\uvicorn.exe app.api:app --reload --host 127.0.0.1 --port 8000
```

Endpoints principais:

- `GET /assets`
- `GET /prices/{ticker}`
- `GET /predictions/{ticker}`
- `GET /predictions/{ticker}/explanation`
- `POST /updates/retrain`
- `POST /paper/signals`
- `GET /paper/signals`
- `GET /paper/blocked`
- `GET /paper/metrics`
- `POST /portfolio/audit`
- `GET /portfolio/positions`
- `GET /portfolio/alerts`

## Deploy de Producao

Arquitetura validada para este projeto:

- frontend Next.js no Vercel
- backend FastAPI em um host Python dedicado

O frontend foi preparado para Vercel, mas o backend atual nao deve ser publicado no Vercel se a expectativa for manter todas as funcionalidades operacionais. O motivo e estrutural: ele depende de SQLite local, artefatos de modelo em disco, `tensorflow`, `torch` e escrita de estado operacional, o que nao combina com o runtime serverless e efemero do Vercel.

Configuracao minima do frontend em producao:

- definir `NEXT_PUBLIC_API_BASE_URL` com a URL publica do backend FastAPI
- liberar CORS no backend com `PROFIT_APP_CORS_ORIGINS=https://seu-frontend.vercel.app`

Sem `NEXT_PUBLIC_API_BASE_URL`, o frontend agora falha de forma explicita em producao para evitar apontar silenciosamente para `localhost`.

## Conselheiro de Risco

Abre posicoes simuladas apenas a partir de teses `simulate_long`, avalia preco atual contra stop, alvo parcial e alvo principal, e salva alertas de risco. Ele nao executa ordens reais.

```powershell
.\.venv311\Scripts\python.exe -m app.cli audit-portfolio
.\.venv311\Scripts\python.exe -m app.cli portfolio-summary
.\.venv311\Scripts\python.exe -m app.cli risk-alert-summary
```

## E2E Completo

```powershell
.\.venv311\Scripts\python.exe tests\e2e_alpha_pipeline.py
.\.venv311\Scripts\python.exe tests\e2e_model_backtest.py
.\.venv311\Scripts\python.exe tests\e2e_paper_news.py
.\.venv311\Scripts\python.exe tests\e2e_qualitative_fusion_api.py
.\.venv311\Scripts\python.exe tests\e2e_dashboard_risk.py
```
