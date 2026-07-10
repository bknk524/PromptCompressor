[CmdletBinding()]
param(
    [string]$FixturePath = "",
    [string]$SettingsDir = "",
    [string]$Profile = "",
    [string[]]$Levels = @("1", "2", "3"),
    [int]$CaseLimit = 0,
    [int]$CaseOffset = 0,
    [string]$LogPath = "",
    [switch]$Trace,
    [switch]$NoDefaultFeatures
)

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Join-Path (Get-Location) "application\tools"
}

$projectRoot = Resolve-Path (Join-Path $scriptRoot "..\..")
if ([string]::IsNullOrWhiteSpace($FixturePath)) {
    $FixturePath = Join-Path $projectRoot "application\resources\evaluations\level-profile-evaluation-v1.json"
}
if ([string]::IsNullOrWhiteSpace($SettingsDir)) {
    $SettingsDir = Join-Path $projectRoot "application\config"
}

if (-not $env:NM_PATH) {
    $candidateNm = "C:\Program Files\LLVM\bin\llvm-nm.exe"
    if (Test-Path $candidateNm) {
        $env:NM_PATH = $candidateNm
    }
}
if (-not $env:OBJCOPY_PATH) {
    $candidateObjcopy = "C:\Program Files\LLVM\bin\llvm-objcopy.exe"
    if (Test-Path $candidateObjcopy) {
        $env:OBJCOPY_PATH = $candidateObjcopy
    }
}
if ($Trace) {
    $env:PROMPT_COMPRESSOR_TRACE = "1"
}
if (-not $env:RUSTFLAGS) {
    $env:RUSTFLAGS = "-Awarnings"
}

$levelValues = @()
foreach ($levelEntry in $Levels) {
    foreach ($levelPart in ($levelEntry -split ",")) {
        $trimmedLevel = $levelPart.Trim()
        if (-not [string]::IsNullOrWhiteSpace($trimmedLevel)) {
            $levelValues += [int]$trimmedLevel
        }
    }
}
if ($levelValues.Count -eq 0) {
    throw "At least one compression level is required."
}

$cargoArgs = @("run", "-q", "-p", "prompt-compressor-cli")
if ($NoDefaultFeatures) {
    $cargoArgs += "--no-default-features"
}
$cargoArgs += "--"
$cargoArgs += @(
    "--settings-dir", $SettingsDir,
    "--eval-fixture", $FixturePath,
    "--eval-levels", ($levelValues -join ","),
    "--eval-progress"
)
if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    $cargoArgs += @("--profile", $Profile)
}
if ($CaseLimit -gt 0) {
    $cargoArgs += @("--eval-case-limit", $CaseLimit)
}
if ($CaseOffset -gt 0) {
    $cargoArgs += @("--eval-case-offset", $CaseOffset)
}

function Join-ProcessArguments {
    param([string[]]$Values)

    ($Values | ForEach-Object {
        $value = [string]$_
        if ($value -match '[\s"]') {
            '"' + $value.Replace('"', '\"') + '"'
        }
        else {
            $value
        }
    }) -join " "
}

Push-Location $projectRoot
try {
    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        & cargo @cargoArgs
    }
    else {
        $resolvedLogPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LogPath)
        $logParent = Split-Path -Parent $resolvedLogPath
        if (-not [string]::IsNullOrWhiteSpace($logParent)) {
            New-Item -ItemType Directory -Force -Path $logParent | Out-Null
        }

        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $processInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processInfo.FileName = "cargo"
        $processInfo.Arguments = Join-ProcessArguments -Values $cargoArgs
        $processInfo.WorkingDirectory = $projectRoot
        $processInfo.UseShellExecute = $false
        $processInfo.RedirectStandardOutput = $true
        $processInfo.RedirectStandardError = $true
        $processInfo.StandardOutputEncoding = $utf8NoBom
        $processInfo.StandardErrorEncoding = $utf8NoBom

        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $processInfo
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()

        [System.IO.File]::WriteAllText($resolvedLogPath, $stdout + $stderr, $utf8NoBom)
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Output $stdout
        }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            if ($process.ExitCode -eq 0) {
                [Console]::Error.WriteLine($stderr)
            }
            else {
                Write-Error $stderr
            }
        }
        exit $process.ExitCode
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
