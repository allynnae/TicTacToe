param(
    [string]$OpenAIKey = $env:OPENAI_API_KEY,
    [string]$WandbKey = $env:WANDB_API_KEY,
    [string]$Seeds = "0,1,2,3",
    [string]$Project = "cs5880-tictactoe",
    [string]$Entity = "am893120",
    [string]$Group = "ppo-tictactoe",
    [int]$TotalTimesteps = 500000,
    [int]$NEnvs = 8
)

if (-not $OpenAIKey) {
    $OpenAIKey = Read-Host "Enter OPENAI_API_KEY (or leave blank to skip)"
}
if (-not $WandbKey) {
    $WandbKey = Read-Host "Enter WANDB_API_KEY (or leave blank for offline mode)"
}

$env:OPENAI_API_KEY = $OpenAIKey
$env:WANDB_API_KEY = $WandbKey

$seedList = $Seeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if (-not $seedList) {
    Write-Error "No seeds provided. Use -Seeds ""0,1,2,3"" for multiple runs."
    exit 1
}

foreach ($seed in $seedList) {
    Write-Host "=== Running seed $seed ===" -ForegroundColor Cyan
    python train_tictactoe_wandb.py `
        --project $Project `
        --entity $Entity `
        --group $Group `
        --total-timesteps $TotalTimesteps `
        --n-envs $NEnvs `
        --seed $seed
}
