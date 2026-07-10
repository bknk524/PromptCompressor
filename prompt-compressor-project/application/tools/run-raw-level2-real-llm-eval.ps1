param(
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$applicationDir = Split-Path -Parent $scriptDir
$projectRoot = Split-Path -Parent $applicationDir

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-Date -Format "yyyyMMdd-HHmmss"
}

$env:NM_PATH = "C:\Program Files\LLVM\bin\llvm-nm.exe"
$env:OBJCOPY_PATH = "C:\Program Files\LLVM\bin\llvm-objcopy.exe"

$cliPath = Join-Path $projectRoot "target\debug\prompt-compressor-cli.exe"
$fixturePath = Join-Path $projectRoot "application\resources\evaluations\raw-prompts-level2-evaluation-v1.json"
$resultDir = Join-Path $projectRoot "application\resources\evaluations\results"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$reportPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.json"
$progressPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.progress.log"
$statusPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.status.json"

if (-not (Test-Path -LiteralPath $cliPath)) {
    throw "prompt-compressor-cli.exe was not found. Build it before running this evaluation: $cliPath"
}

if (-not (Test-Path -LiteralPath $fixturePath)) {
    throw "Evaluation fixture was not found: $fixturePath"
}

$startedAt = (Get-Date).ToString("o")
@{
    run_id = $RunId
    status = "running"
    started_at = $startedAt
    profile = "internal_llm"
    runtime = "llama_cpp_embedded"
    fixture = $fixturePath
    report = $reportPath
    progress_log = $progressPath
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $statusPath

$cliArguments = "--profile internal_llm --eval-fixture `"$fixturePath`" --eval-levels 2 --eval-progress"
$cliProcess = Start-Process `
    -FilePath $cliPath `
    -ArgumentList $cliArguments `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru `
    -Wait `
    -RedirectStandardOutput $reportPath `
    -RedirectStandardError $progressPath

$exitCode = $cliProcess.ExitCode
$finishedAt = (Get-Date).ToString("o")
$finalStatus = if ($exitCode -eq 0) { "finished" } else { "failed" }

@{
    run_id = $RunId
    status = $finalStatus
    exit_code = $exitCode
    started_at = $startedAt
    finished_at = $finishedAt
    profile = "internal_llm"
    runtime = "llama_cpp_embedded"
    fixture = $fixturePath
    report = $reportPath
    progress_log = $progressPath
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $statusPath

exit $exitCode
