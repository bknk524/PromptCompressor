[CmdletBinding()]
param(
    [string]$RunId = "",
    [string]$FixturePath = "",
    [string]$Profile = "internal_llm",
    [string[]]$Levels = @("2"),
    [int]$CaseLimit = 0,
    [int]$CaseOffset = 0,
    [ValidateSet("raw-model", "final-pipeline")]
    [string]$EvaluationStage = "raw-model"
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
if ($RunId -notmatch '^[A-Za-z0-9._-]+$') {
    throw "RunId may contain only letters, numbers, dots, underscores, and hyphens."
}
if ($CaseLimit -lt 0 -or $CaseOffset -lt 0) {
    throw "CaseLimit and CaseOffset must be zero or greater."
}

if ([string]::IsNullOrWhiteSpace($FixturePath)) {
    $FixturePath = Join-Path $projectRoot "application\resources\evaluations\raw-prompts-level2-evaluation-v1.json"
} elseif (-not [System.IO.Path]::IsPathRooted($FixturePath)) {
    $FixturePath = Join-Path $projectRoot $FixturePath
}
$FixturePath = [System.IO.Path]::GetFullPath($FixturePath)
if (-not (Test-Path -LiteralPath $FixturePath -PathType Leaf)) {
    throw "Evaluation fixture was not found: $FixturePath"
}

$levelValues = @()
foreach ($levelEntry in $Levels) {
    foreach ($levelPart in ($levelEntry -split ',')) {
        $trimmed = $levelPart.Trim()
        if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
            $level = [int]$trimmed
            if ($level -lt 1 -or $level -gt 3) {
                throw "Evaluation levels must be between 1 and 3."
            }
            $levelValues += $level
        }
    }
}
if ($levelValues.Count -eq 0) {
    throw "At least one evaluation level is required."
}
$levelValues = @($levelValues | Select-Object -Unique)

function Set-LlvmToolPath {
    param(
        [Parameter(Mandatory = $true)][string]$VariableName,
        [Parameter(Mandatory = $true)][string[]]$Candidates
    )

    $current = [Environment]::GetEnvironmentVariable($VariableName, "Process")
    if (-not [string]::IsNullOrWhiteSpace($current) -and (Test-Path -LiteralPath $current -PathType Leaf)) {
        return
    }
    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            [Environment]::SetEnvironmentVariable($VariableName, $candidate, "Process")
            return
        }
    }
    throw "$VariableName could not be resolved. Install LLVM or set the variable to the required executable."
}

$rustSysroot = (& rustc --print sysroot | Select-Object -First 1).Trim()
$rustHostLine = & rustc -vV | Where-Object { $_ -match '^host:\s*(.+)$' } | Select-Object -First 1
$rustHost = if ($rustHostLine -match '^host:\s*(.+)$') { $Matches[1].Trim() } else { "" }
if ([string]::IsNullOrWhiteSpace($rustSysroot) -or [string]::IsNullOrWhiteSpace($rustHost)) {
    throw "The active Rust toolchain could not be resolved."
}
$rustLlvmBin = if ($rustSysroot -and $rustHost) {
    Join-Path $rustSysroot "lib\rustlib\$rustHost\bin"
} else {
    ""
}
Set-LlvmToolPath -VariableName "NM_PATH" -Candidates @(
    "C:\Program Files\LLVM\bin\llvm-nm.exe",
    (Join-Path $rustLlvmBin "llvm-nm.exe")
)
Set-LlvmToolPath -VariableName "OBJCOPY_PATH" -Candidates @(
    "C:\Program Files\LLVM\bin\llvm-objcopy.exe",
    (Join-Path $rustLlvmBin "llvm-objcopy.exe")
)

$libclangPath = [Environment]::GetEnvironmentVariable("LIBCLANG_PATH", "Process")
if ([string]::IsNullOrWhiteSpace($libclangPath) -or -not (Test-Path -LiteralPath (Join-Path $libclangPath "libclang.dll") -PathType Leaf)) {
    foreach ($candidate in @("C:\Program Files\LLVM\bin", "C:\Program Files (x86)\LLVM\bin")) {
        if (Test-Path -LiteralPath (Join-Path $candidate "libclang.dll") -PathType Leaf) {
            $env:LIBCLANG_PATH = $candidate
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($env:LIBCLANG_PATH) -or -not (Test-Path -LiteralPath (Join-Path $env:LIBCLANG_PATH "libclang.dll") -PathType Leaf)) {
    throw "LIBCLANG_PATH could not be resolved. Install LLVM or set it to the directory containing libclang.dll."
}

$resultDir = Join-Path $projectRoot "application\resources\evaluations\results"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$reportPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.json"
$progressPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.progress.log"
$statusPath = Join-Path $resultDir "raw-level2-real-llm-$RunId.status.json"

function Write-RunStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$StartedAt,
        [Nullable[int]]$ExitCode,
        [object]$Report,
        [string]$ErrorMessage = ""
    )

    $statusValue = [ordered]@{
        run_id = $RunId
        status = $Status
        started_at = $StartedAt
        profile = $Profile
        levels = $levelValues
        case_limit = $CaseLimit
        case_offset = $CaseOffset
        evaluation_stage = $EvaluationStage
        fixture = $FixturePath
        report = $reportPath
        progress_log = $progressPath
    }
    if ($null -ne $ExitCode) {
        $statusValue.exit_code = $ExitCode
        $statusValue.finished_at = (Get-Date).ToString("o")
    }
    if ($null -ne $Report) {
        $statusValue.passed = [bool]$Report.passed
        $statusValue.case_count = [int]$Report.case_count
        $statusValue.run_count = [int]$Report.run_count
        $statusValue.failure_count = [int]$Report.failure_count
    }
    if (-not [string]::IsNullOrWhiteSpace($ErrorMessage)) {
        $statusValue.error = $ErrorMessage
    }
    $statusValue | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath $statusPath
}

function Join-ProcessArguments {
    param([string[]]$Values)

    ($Values | ForEach-Object {
        $value = [string]$_
        if ($value -match '[\s"]') {
            '"' + $value.Replace('"', '\"') + '"'
        } else {
            $value
        }
    }) -join ' '
}

$settingsDir = Join-Path $projectRoot "application\config"
$cargoArguments = @(
    "run", "-q", "-p", "prompt-compressor-cli", "--",
    "--settings-dir", $settingsDir,
    "--profile", $Profile,
    "--eval-fixture", $FixturePath,
    "--eval-levels", ($levelValues -join ','),
    "--eval-stage", $EvaluationStage,
    "--eval-progress"
)
if ($CaseLimit -gt 0) {
    $cargoArguments += @("--eval-case-limit", $CaseLimit)
}
if ($CaseOffset -gt 0) {
    $cargoArguments += @("--eval-case-offset", $CaseOffset)
}

$startedAt = (Get-Date).ToString("o")
Write-RunStatus -Status "running" -StartedAt $startedAt

try {
    $process = Start-Process `
        -FilePath "cargo" `
        -ArgumentList (Join-ProcessArguments -Values $cargoArguments) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru `
        -Wait `
        -RedirectStandardOutput $reportPath `
        -RedirectStandardError $progressPath
    $exitCode = $process.ExitCode
} catch {
    Write-RunStatus -Status "failed" -StartedAt $startedAt -ExitCode 1 -ErrorMessage $_.Exception.Message
    throw
}

$report = $null
$reportError = ""
try {
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "Evaluation report was not created."
    }
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    $reportError = $_.Exception.Message
    if ($exitCode -eq 0) {
        $exitCode = 1
    }
}

$finalStatus = if ($exitCode -eq 0 -and $null -ne $report -and $report.passed) { "finished" } else { "failed" }
Write-RunStatus -Status $finalStatus -StartedAt $startedAt -ExitCode $exitCode -Report $report -ErrorMessage $reportError
exit $exitCode
