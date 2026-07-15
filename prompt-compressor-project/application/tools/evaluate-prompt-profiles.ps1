[CmdletBinding()]
param(
    [string]$FixturePath = "",
    [string]$SettingsDir = "",
    [string]$Profile = "",
    [string[]]$Levels = @("1", "2", "3"),
    [int]$CaseLimit = 0,
    [int]$CaseOffset = 0,
    [string]$LogPath = "",
    [string]$ProgressPath = "",
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
$rustLlvmBin = Join-Path $rustSysroot "lib\rustlib\$rustHost\bin"
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
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()

        [System.IO.File]::WriteAllText($resolvedLogPath, $stdout, $utf8NoBom)
        if (-not [string]::IsNullOrWhiteSpace($ProgressPath)) {
            $resolvedProgressPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProgressPath)
            $progressParent = Split-Path -Parent $resolvedProgressPath
            if (-not [string]::IsNullOrWhiteSpace($progressParent)) {
                New-Item -ItemType Directory -Force -Path $progressParent | Out-Null
            }
            [System.IO.File]::WriteAllText($resolvedProgressPath, $stderr, $utf8NoBom)
        }
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Output $stdout
        }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            [Console]::Error.WriteLine($stderr)
        }
        exit $process.ExitCode
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
