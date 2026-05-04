param(
    [string]$PythonPath = ".\.venv311\Scripts\python.exe",
    [string]$TfRunId = "",
    [string]$TradeOutcomeRunId = "",
    [double]$PortfolioValue = 10000.0,
    [double]$MaxRiskPerTrade = 0.01,
    [double]$MinRewardRiskRatio = 1.5,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) {
    throw "Python not found at $PythonPath"
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host "`n==> $Name" -ForegroundColor Cyan
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

Invoke-Step "Generate technical features" @("-m", "app.cli", "generate-features")

if (-not $SkipTraining) {
    Invoke-Step "Train TensorFlow direction model" @("-m", "app.cli", "train-tf-direction", "--epochs", "40", "--batch-size", "64")
    Invoke-Step "Train trade-outcome model" @("-m", "app.cli", "train-trade-outcome", "--holding-days", "7", "--min-reward-risk", "1.5", "--cost-per-trade", "0.002", "--spread", "0.001", "--slippage", "0.001", "--max-iter", "160", "--learning-rate", "0.05")
}

$tfInferenceArgs = @("-m", "app.cli", "run-current-inference")
if ($TfRunId) { $tfInferenceArgs += @("--run-id", $TfRunId) }
Invoke-Step "Run TensorFlow current inference" $tfInferenceArgs

$fusionArgs = @("-m", "app.cli", "run-fusion")
if ($TfRunId) { $fusionArgs += @("--run-id", $TfRunId) }
Invoke-Step "Run fusion" $fusionArgs

$tradeOutcomeArgs = @("-m", "app.cli", "run-trade-outcome-inference")
if ($TradeOutcomeRunId) { $tradeOutcomeArgs += @("--run-id", $TradeOutcomeRunId) }
Invoke-Step "Run trade-outcome inference" $tradeOutcomeArgs

$paperArgs = @(
    "-m", "app.cli", "generate-paper-signals",
    "--portfolio-value", $PortfolioValue.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--max-risk-per-trade", $MaxRiskPerTrade.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--min-reward-risk-ratio", $MinRewardRiskRatio.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($TfRunId) { $paperArgs += @("--run-id", $TfRunId) }
Invoke-Step "Generate paper signals" $paperArgs

Write-Host "`nML pipeline completed." -ForegroundColor Green
