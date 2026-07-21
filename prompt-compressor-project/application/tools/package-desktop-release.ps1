param(
    [string]$OutputPath,
    [switch]$SkipBuild,
    [switch]$Clean,
    [switch]$Zip,
    [switch]$SkipLocalModelSmokeTest
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptRoot "..\..")
$projectParent = Split-Path -Parent $projectRoot
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $packagePath = Join-Path $projectParent "TrimPrompt-exe"
} elseif ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $packagePath = $OutputPath
} else {
    $packagePath = Join-Path $projectRoot $OutputPath
}
$packagePath = [System.IO.Path]::GetFullPath($packagePath)
$outputRootPath = Split-Path -Parent $packagePath
$applicationPath = Join-Path $projectRoot "application"
$configPath = Join-Path $applicationPath "config"
$resourcesPath = Join-Path $applicationPath "resources"
$launcherReleaseExe = Join-Path $projectRoot "target\release\trim-prompt-launcher.exe"
$buildIdPath = Join-Path $projectRoot "target\trimprompt-build-id.txt"
$inferenceCompatibilityIdPath = Join-Path $projectRoot "target\trimprompt-inference-compatibility-id.txt"
$cpuTargetTriple = "x86_64-pc-windows-msvc"
$avx2TargetPath = Join-Path $projectRoot "target\cpu-avx2"
$avx2ReleaseExe = Join-Path $avx2TargetPath "$cpuTargetTriple\release\prompt-compressor-desktop.exe"
$avx512TargetPath = Join-Path $projectRoot "target\cpu-avx512"
$avx512ReleaseExe = Join-Path $avx512TargetPath "$cpuTargetTriple\release\prompt-compressor-desktop.exe"
$compatibleTargetPath = Join-Path $projectRoot "target\cpu-compatible"
$compatibleReleaseExe = Join-Path $compatibleTargetPath "$cpuTargetTriple\release\prompt-compressor-desktop.exe"
$packageExe = Join-Path $packagePath "TrimPrompt.exe"
$packageCpuRuntimePath = Join-Path $packagePath "application\runtime\cpu"
$packageAvx2Exe = Join-Path $packageCpuRuntimePath "TrimPrompt-avx2.exe"
$packageAvx512Exe = Join-Path $packageCpuRuntimePath "TrimPrompt-avx512.exe"
$packageCompatibleExe = Join-Path $packageCpuRuntimePath "TrimPrompt-compatible.exe"

function Get-InferenceCompatibilityId {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    $inputs = @(
        "Cargo.toml",
        "Cargo.lock",
        "application\core\compression-core\Cargo.toml",
        "application\core\compression-core\src",
        "application\ui\web\Cargo.toml",
        "application\ui\web\src",
        "application\ui\desktop\Cargo.toml",
        "application\ui\desktop\src\main.rs",
        "application\config",
        "application\resources\prompts",
        "application\vendor\llama-cpp-sys-4"
    )
    $files = foreach ($relativePath in $inputs) {
        $path = Join-Path $ProjectRoot $relativePath
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Get-Item -LiteralPath $path
        } elseif (Test-Path -LiteralPath $path -PathType Container) {
            Get-ChildItem -LiteralPath $path -Recurse -File
        } else {
            throw "Inference compatibility input is missing: $path"
        }
    }

    $manifestLines = foreach ($file in ($files | Sort-Object -Property FullName -Unique)) {
        $relativePath = $file.FullName.Substring($ProjectRoot.Length).TrimStart('\').Replace('\', '/')
        $fileHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$relativePath|$fileHash"
    }
    $payload = "trimprompt-inference-compatibility-v1`n$($manifestLines -join "`n")"
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($payload))
    } finally {
        $sha256.Dispose()
    }
    return [System.BitConverter]::ToString($digest).Replace("-", "").ToLowerInvariant()
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Child
    )

    $parentFull = [System.IO.Path]::GetFullPath($Parent)
    $childFull = [System.IO.Path]::GetFullPath($Child)
    if (-not $childFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside ${parentFull}: $childFull"
    }
}

function Copy-Directory {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path $Source)) {
        throw "Required directory not found: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($entry in Get-ChildItem -LiteralPath $Source -Force) {
        $entryDestination = Join-Path $Destination $entry.Name
        if ($entry.PSIsContainer) {
            Copy-Directory -Source $entry.FullName -Destination $entryDestination
        } else {
            Copy-Item -LiteralPath $entry.FullName -Destination $entryDestination -Force
        }
    }
}

function Remove-PackageEntry {
    param(
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [Parameter(Mandatory = $true)][string]$Path
    )

    Assert-ChildPath -Parent $PackageRoot -Child $Path
    $item = Get-Item -LiteralPath $Path -Force
    $isReparsePoint = ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            if ($isReparsePoint) {
                Remove-Item -LiteralPath $Path -Force
            } else {
                Remove-Item -LiteralPath $Path -Recurse -Force
            }
            return
        } catch {
            if ($attempt -eq 19) {
                throw
            }
            Start-Sleep -Milliseconds 500
        }
    }
}

function Reset-PackageManagedContent {
    param([Parameter(Mandatory = $true)][string]$PackagePath)

    $packageItem = Get-Item -LiteralPath $PackagePath -Force
    if (($packageItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Package path must not be a reparse point: $PackagePath"
    }

    # Preserve machine-local models and state while refreshing managed package files.
    foreach ($entry in Get-ChildItem -LiteralPath $PackagePath -Force) {
        if ($entry.Name -ieq "application" -and $entry.PSIsContainer) {
            continue
        }
        Remove-PackageEntry -PackageRoot $PackagePath -Path $entry.FullName
    }

    $packageApplicationPath = Join-Path $PackagePath "application"
    if (-not (Test-Path -LiteralPath $packageApplicationPath -PathType Container)) {
        return
    }
    $applicationItem = Get-Item -LiteralPath $packageApplicationPath -Force
    if (($applicationItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Package application path must not be a reparse point: $packageApplicationPath"
    }

    foreach ($entry in Get-ChildItem -LiteralPath $packageApplicationPath -Force) {
        if ($entry.Name -ieq "local" -and $entry.PSIsContainer) {
            continue
        }
        if ($entry.Name -ieq "config" -and $entry.PSIsContainer) {
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Package config path must not be a reparse point: $($entry.FullName)"
            }
            foreach ($configEntry in Get-ChildItem -LiteralPath $entry.FullName -Force) {
                if ($configEntry.Name -ieq "user" -and $configEntry.PSIsContainer) {
                    continue
                }
                Remove-PackageEntry -PackageRoot $PackagePath -Path $configEntry.FullName
            }
            continue
        }
        Remove-PackageEntry -PackageRoot $PackagePath -Path $entry.FullName
    }
}

function Copy-RelativeFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath
    )

    $source = Join-Path $applicationPath $RelativePath
    if (-not (Test-Path $source)) {
        throw "Required file not found: $source"
    }

    $destination = Join-Path (Join-Path $packagePath "application") $RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Use-LlvmBuildTools {
    $candidateBins = @()
    if (-not [string]::IsNullOrWhiteSpace($env:LIBCLANG_PATH)) {
        $candidateBins += $env:LIBCLANG_PATH
    }
    $candidateBins += @(
        "C:\Program Files\LLVM\bin",
        "C:\Program Files (x86)\LLVM\bin"
    )

    foreach ($bin in $candidateBins) {
        if ([string]::IsNullOrWhiteSpace($bin)) {
            continue
        }
        $libclang = Join-Path $bin "libclang.dll"
        $llvmNm = Join-Path $bin "llvm-nm.exe"
        if ((Test-Path $libclang) -and (Test-Path $llvmNm)) {
            $env:LIBCLANG_PATH = $bin
            $pathParts = $env:PATH -split ';'
            if (-not ($pathParts -contains $bin)) {
                $env:PATH = "$bin;$env:PATH"
            }
            return
        }
    }

    throw "LLVM build tools were not found. Install LLVM, or set LIBCLANG_PATH to the folder containing libclang.dll and llvm-nm.exe."
}

function Build-CpuRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CargoFeature,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$RustTargetFeatures
    )

    $previousEncodedRustFlags = $env:CARGO_ENCODED_RUSTFLAGS
    $previousRustFlags = $env:RUSTFLAGS
    try {
        # --targetを付けることで、AVX-512非対応のビルドPCでもbuild.rsは汎用命令のまま実行する。
        # encoded形式ならPowerShellやCargoによる命令名の引数分割も避けられる。
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        $env:CARGO_ENCODED_RUSTFLAGS = "-Ctarget-feature=$RustTargetFeatures"
        Write-Host "Building $Name CPU runtime (Rust: $RustTargetFeatures)..."
        cargo build `
            --release `
            --target $cpuTargetTriple `
            -p prompt-compressor-desktop `
            --no-default-features `
            --features "$CargoFeature,cpu-profile-strict" `
            --target-dir $TargetPath
        if ($LASTEXITCODE -ne 0) {
            throw "$Name CPU runtime build failed with exit code $LASTEXITCODE"
        }
    } finally {
        if ($null -eq $previousRustFlags) {
            Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        } else {
            $env:RUSTFLAGS = $previousRustFlags
        }
        if ($null -eq $previousEncodedRustFlags) {
            Remove-Item Env:CARGO_ENCODED_RUSTFLAGS -ErrorAction SilentlyContinue
        } else {
            $env:CARGO_ENCODED_RUSTFLAGS = $previousEncodedRustFlags
        }
    }
}

function Stop-PackageProcesses {
    param([Parameter(Mandatory = $true)][string]$PackagePath)

    $packageFull = [System.IO.Path]::GetFullPath($PackagePath)
    $desktopExecutableNames = @(
        "TrimPrompt.exe",
        "TrimPrompt-avx2.exe",
        "TrimPrompt-avx512.exe",
        "TrimPrompt-compatible.exe",
        "PromptCompressor.exe"
    )
    $processes = Get-CimInstance Win32_Process | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine.IndexOf($packageFull, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        ($desktopExecutableNames -contains $_.Name -or $_.Name -eq "msedgewebview2.exe")
    }

    $desktopProcesses = $processes | Where-Object { $desktopExecutableNames -contains $_.Name }
    foreach ($process in $desktopProcesses) {
        Write-Host "Stopping running packaged app process: $($process.ProcessId)"
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    if ($desktopProcesses) {
        Start-Sleep -Seconds 2
    }

    $remainingWebViews = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "msedgewebview2.exe" -and
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine.IndexOf($packageFull, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    }
    foreach ($process in $remainingWebViews) {
        Write-Host "Stopping packaged WebView2 helper process: $($process.ProcessId)"
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Remove-PackagedModel {
    param(
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][string]$ModelPath
    )

    Stop-PackageProcesses -PackagePath $PackagePath
    foreach ($path in @($ModelPath, "$ModelPath.verified.json")) {
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $stream = $null
            if (-not (Test-Path -LiteralPath $path)) {
                break
            }
            try {
                $stream = [System.IO.File]::Open(
                    $path,
                    [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::ReadWrite,
                    [System.IO.FileShare]::None
                )
                $stream.Dispose()
                [System.IO.File]::Delete($path)
                break
            } catch {
                if ($null -ne $stream) {
                    $stream.Dispose()
                }
                if ($attempt -eq 19) {
                    throw
                }
                Start-Sleep -Milliseconds 500
            }
        }
    }
}

function Assert-PackagedFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [long]$MinimumBytes = 1
    )

    $path = Join-Path $packagePath $RelativePath
    if (-not (Test-Path $path)) {
        throw "Packaged file is missing: $path"
    }

    $item = Get-Item -LiteralPath $path
    if ($item.Length -lt $MinimumBytes) {
        throw "Packaged file is too small: $path ($($item.Length) bytes)"
    }
}

function Invoke-PackagedCpuEngine {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][string]$Engine,
        [Parameter(Mandatory = $true)][string]$BuildId,
        [Parameter(Mandatory = $true)][string]$InferenceCompatibilityId,
        [Parameter(Mandatory = $true)][string]$Argument
    )

    $previousEngine = [System.Environment]::GetEnvironmentVariable("TRIMPROMPT_CPU_ENGINE", "Process")
    $previousBuildId = [System.Environment]::GetEnvironmentVariable("TRIMPROMPT_EXPECTED_BUILD_ID", "Process")
    $previousCompatibilityId = [System.Environment]::GetEnvironmentVariable("TRIMPROMPT_INFERENCE_COMPATIBILITY_ID", "Process")
    try {
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_CPU_ENGINE", $Engine, "Process")
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_EXPECTED_BUILD_ID", $BuildId, "Process")
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_INFERENCE_COMPATIBILITY_ID", $InferenceCompatibilityId, "Process")
        return Start-Process `
            -FilePath $ExecutablePath `
            -WorkingDirectory $PackagePath `
            -ArgumentList $Argument `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
    } finally {
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_CPU_ENGINE", $previousEngine, "Process")
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_EXPECTED_BUILD_ID", $previousBuildId, "Process")
        [System.Environment]::SetEnvironmentVariable("TRIMPROMPT_INFERENCE_COMPATIBILITY_ID", $previousCompatibilityId, "Process")
    }
}

function Test-PackagedCpuEngineSupport {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][string]$Engine,
        [Parameter(Mandatory = $true)][string]$BuildId,
        [Parameter(Mandatory = $true)][string]$InferenceCompatibilityId
    )

    $process = Invoke-PackagedCpuEngine `
        -ExecutablePath $ExecutablePath `
        -PackagePath $PackagePath `
        -Engine $Engine `
        -BuildId $BuildId `
        -InferenceCompatibilityId $InferenceCompatibilityId `
        -Argument "--cpu-engine-support-probe"
    return $process.ExitCode -eq 0
}

function Test-PackagedLocalModelCompression {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][string]$Engine,
        [Parameter(Mandatory = $true)][string]$BuildId,
        [Parameter(Mandatory = $true)][string]$InferenceCompatibilityId
    )

    Write-Host "Running packaged $Engine local model quality test..."
    $resultPath = Join-Path $PackagePath "application\local\state\package-smoke-result.json"
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
        throw "Packaged executable is missing: $ExecutablePath"
    }
    if (Test-Path -LiteralPath $resultPath) {
        Remove-Item -LiteralPath $resultPath -Force
    }

    $json = $null
    try {
        $process = Invoke-PackagedCpuEngine `
            -ExecutablePath $ExecutablePath `
            -PackagePath $PackagePath `
            -Engine $Engine `
            -BuildId $BuildId `
            -InferenceCompatibilityId $InferenceCompatibilityId `
            -Argument "--package-smoke-test"
        if (Test-Path -LiteralPath $resultPath -PathType Leaf) {
            $json = (Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8).Trim()
        }
        if ($process.ExitCode -ne 0) {
            throw "Packaged executable smoke test failed with exit code $($process.ExitCode): $json"
        }
    } finally {
        if (Test-Path -LiteralPath $resultPath) {
            Remove-Item -LiteralPath $resultPath -Force
        }
    }

    if ([string]::IsNullOrWhiteSpace($json)) {
        throw "Packaged local model smoke test returned no JSON"
    }

    $result = $json | ConvertFrom-Json
    if ($result.profile -ne "internal_llm") {
        throw "Packaged local model smoke test used unexpected profile: $($result.profile)"
    }
    if ($result.runtime -ne "llama_cpp_embedded") {
        throw "Packaged local model smoke test used unexpected runtime: $($result.runtime)"
    }
    if ($result.cpu_engine -ne $Engine) {
        throw "Packaged local model smoke test used unexpected CPU engine: $($result.cpu_engine)"
    }
    if (-not $result.quality_passed -or $result.quality_case_count -ne 5) {
        throw "Packaged local model quality gate did not pass all five cases"
    }
    if ($result.should_send_original) {
        throw "Packaged local model smoke test returned original prompt: $($result.fallback_reason)"
    }
    if ([string]::IsNullOrWhiteSpace($result.distilled_prompt)) {
        throw "Packaged local model smoke test returned an empty compressed prompt"
    }
    if ($result.metrics.output_characters -ge $result.metrics.input_characters) {
        throw "Packaged local model smoke test did not reduce character count"
    }

    Write-Host "Packaged $Engine local model quality test passed:"
    Write-Host $result.distilled_prompt
}

function Read-ProfileValue {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $value = Try-Read-ProfileValue -Path $Path -ProfileId $ProfileId -Key $Key
    if ($null -ne $value) {
        return $value
    }

    throw "Key '$Key' not found for '$ProfileId' in $Path"
}

function Try-Read-ProfileValue {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$Key
    )

    $lines = Get-Content -Encoding UTF8 $Path
    $inside = $false
    foreach ($line in $lines) {
        if ($line -match "^\s{2}$([regex]::Escape($ProfileId)):\s*$") {
            $inside = $true
            continue
        }
        if ($inside -and $line -match "^\s{2}\S") {
            break
        }
        if ($inside -and $line -match "^\s{4}$([regex]::Escape($Key)):\s*(.+?)\s*$") {
            return $Matches[1].Trim("'`"")
        }
    }

    return $null
}

function Read-DefaultProfile {
    param([Parameter(Mandatory = $true)][string]$Path)

    $line = Get-Content -Encoding UTF8 $Path | Where-Object { $_ -match "^\s*default_profile:\s*(.+?)\s*$" } | Select-Object -First 1
    if (-not $line) {
        throw "default_profile not found in $Path"
    }
    if ($line -match "^\s*default_profile:\s*(.+?)\s*$") {
        return $Matches[1].Trim("'`"")
    }
    throw "default_profile could not be parsed in $Path"
}

if (-not $SkipBuild) {
    $gitRevision = (& git -C $projectRoot rev-parse --short=12 HEAD 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($gitRevision)) {
        $gitRevision = "unknown"
    }
    $buildTimestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddHHmmss")
    $buildId = "$buildTimestamp-$gitRevision"
    $inferenceCompatibilityId = Get-InferenceCompatibilityId -ProjectRoot $projectRoot
    $env:TRIMPROMPT_BUILD_ID = $buildId
    $env:TRIMPROMPT_INFERENCE_COMPATIBILITY_ID = $inferenceCompatibilityId
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $buildIdPath) | Out-Null
    $buildId | Set-Content -LiteralPath $buildIdPath -Encoding ASCII
    $inferenceCompatibilityId | Set-Content -LiteralPath $inferenceCompatibilityIdPath -Encoding ASCII

    Push-Location $projectRoot
    try {
        Use-LlvmBuildTools
        cargo build --release -p trim-prompt-launcher
        if ($LASTEXITCODE -ne 0) {
            throw "launcher build failed with exit code $LASTEXITCODE"
        }
        Build-CpuRuntime `
            -Name "SSE4.2 compatible" `
            -CargoFeature "embedded-llama-compatible" `
            -TargetPath $compatibleTargetPath `
            -RustTargetFeatures "+sse4.2"
        Build-CpuRuntime `
            -Name "AVX2" `
            -CargoFeature "embedded-llama-avx2" `
            -TargetPath $avx2TargetPath `
            -RustTargetFeatures "+sse4.2,+avx,+avx2,+fma,+f16c,+bmi2"
        Build-CpuRuntime `
            -Name "AVX-512" `
            -CargoFeature "embedded-llama-avx512" `
            -TargetPath $avx512TargetPath `
            -RustTargetFeatures "+sse4.2,+avx,+avx2,+fma,+f16c,+bmi2,+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl"
    } finally {
        Pop-Location
    }
} else {
    if (-not (Test-Path -LiteralPath $buildIdPath -PathType Leaf)) {
        throw "Build ID was not found for -SkipBuild: $buildIdPath"
    }
    if (-not (Test-Path -LiteralPath $inferenceCompatibilityIdPath -PathType Leaf)) {
        throw "Inference compatibility ID was not found for -SkipBuild: $inferenceCompatibilityIdPath"
    }
    $buildId = (Get-Content -LiteralPath $buildIdPath -Raw -Encoding ASCII).Trim()
    $inferenceCompatibilityId = (Get-Content -LiteralPath $inferenceCompatibilityIdPath -Raw -Encoding ASCII).Trim()
}

foreach ($requiredExecutable in @(
    $launcherReleaseExe,
    $avx2ReleaseExe,
    $avx512ReleaseExe,
    $compatibleReleaseExe
)) {
    if (-not (Test-Path $requiredExecutable)) {
        throw "Release executable not found: $requiredExecutable"
    }
}

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
Assert-ChildPath -Parent $outputRootPath -Child $packagePath

if (Test-Path $packagePath) {
    if (-not $Clean) {
        throw "Package path already exists. Re-run with -Clean to replace it: $packagePath"
    }
    Stop-PackageProcesses -PackagePath $packagePath
    Reset-PackageManagedContent -PackagePath $packagePath
}

New-Item -ItemType Directory -Force -Path $packagePath | Out-Null
New-Item -ItemType Directory -Force -Path $packageCpuRuntimePath | Out-Null
Copy-Item -LiteralPath $launcherReleaseExe -Destination $packageExe -Force
Copy-Item -LiteralPath $avx2ReleaseExe -Destination $packageAvx2Exe -Force
Copy-Item -LiteralPath $avx512ReleaseExe -Destination $packageAvx512Exe -Force
Copy-Item -LiteralPath $compatibleReleaseExe -Destination $packageCompatibleExe -Force
Assert-PackagedFile -RelativePath "TrimPrompt.exe" -MinimumBytes 1024
Assert-PackagedFile -RelativePath "application\runtime\cpu\TrimPrompt-avx2.exe" -MinimumBytes 1048576
Assert-PackagedFile -RelativePath "application\runtime\cpu\TrimPrompt-avx512.exe" -MinimumBytes 1048576
Assert-PackagedFile -RelativePath "application\runtime\cpu\TrimPrompt-compatible.exe" -MinimumBytes 1048576

Copy-Directory -Source $configPath -Destination (Join-Path $packagePath "application\config")
Copy-Directory `
    -Source (Join-Path $resourcesPath "prompts") `
    -Destination (Join-Path $packagePath "application\resources\prompts")

$profilesPath = Join-Path $configPath "compression-profiles.yaml"
$modelsPath = Join-Path $configPath "model-catalog.yaml"
$runtimesPath = Join-Path $configPath "runtime-backends.yaml"
$defaultProfile = Read-DefaultProfile -Path $profilesPath
$modelRef = Read-ProfileValue -Path $profilesPath -ProfileId $defaultProfile -Key "model_ref"
$runtimeRef = Read-ProfileValue -Path $profilesPath -ProfileId $defaultProfile -Key "runtime_ref"
$modelPath = Read-ProfileValue -Path $modelsPath -ProfileId $modelRef -Key "model_path"
$runtimeExecutablePath = Try-Read-ProfileValue -Path $runtimesPath -ProfileId $runtimeRef -Key "executable_path"

$packagedModelPath = Join-Path (Join-Path $packagePath "application") $modelPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $packagedModelPath) | Out-Null

if (-not [string]::IsNullOrWhiteSpace($runtimeExecutablePath)) {
    $runtimeAssetPath = $runtimeExecutablePath
} else {
    $runtimeAssetPath = $null
}

if (-not [string]::IsNullOrWhiteSpace($runtimeAssetPath)) {
    $runtimeRelativeDirectory = Split-Path -Parent $runtimeAssetPath
    $runtimeSourceDirectory = Join-Path $applicationPath $runtimeRelativeDirectory
    $runtimeDestinationDirectory = Join-Path (Join-Path $packagePath "application") $runtimeRelativeDirectory
    Copy-Directory -Source $runtimeSourceDirectory -Destination $runtimeDestinationDirectory
    $runtimeManifest = "application/$runtimeRelativeDirectory"
} else {
    $runtimeManifest = "$runtimeRef (embedded in executable)"
}

$localPath = Join-Path $packagePath "application\local"
foreach ($dir in @("cache", "logs", "state")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $localPath $dir) | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $packagePath "application\config\user") | Out-Null

if (Test-Path (Join-Path $projectRoot "README.md")) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "README.md") -Destination (Join-Path $packagePath "README.md") -Force
}

$manifest = @"
TrimPrompt desktop package

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Build ID: $buildId
Inference compatibility ID: $inferenceCompatibilityId
Executable: TrimPrompt.exe
Default profile: $defaultProfile
Model: downloaded from Hugging Face on first launch to application/$modelPath
Runtime: $runtimeManifest; CPU dispatch selects AVX-512, AVX2, or compatible engine at startup
CPU builds: Rust and llama.cpp use matching SSE4.2, AVX2, or AVX-512 instruction profiles
Desktop transport: WebView2 custom protocol (no HTTP server, no localhost port)
Update behavior: application/local and application/config/user are preserved by -Clean

This folder is a runnable package. It is intentionally separate from the source
project folder and contains only the desktop executable plus runtime assets.
"@
$manifest | Set-Content -Encoding UTF8 (Join-Path $packagePath "PACKAGE_MANIFEST.txt")

if (-not $SkipLocalModelSmokeTest) {
    $preservedModel = Test-Path -LiteralPath $packagedModelPath -PathType Leaf
    if (-not $preservedModel) {
        Copy-RelativeFile -RelativePath $modelPath
    }
    Assert-PackagedFile -RelativePath "application\$modelPath" -MinimumBytes 1048576
    try {
        $cpuEngines = @(
            [PSCustomObject]@{ Name = "compatible"; Executable = $packageCompatibleExe },
            [PSCustomObject]@{ Name = "avx2"; Executable = $packageAvx2Exe },
            [PSCustomObject]@{ Name = "avx512"; Executable = $packageAvx512Exe }
        )
        $testedCpuEngineCount = 0
        foreach ($cpuEngine in $cpuEngines) {
            $supported = Test-PackagedCpuEngineSupport `
                -ExecutablePath $cpuEngine.Executable `
                -PackagePath $packagePath `
                -Engine $cpuEngine.Name `
                -BuildId $buildId `
                -InferenceCompatibilityId $inferenceCompatibilityId
            if (-not $supported) {
                Write-Host "Skipping unsupported packaged CPU engine: $($cpuEngine.Name)"
                continue
            }
            Test-PackagedLocalModelCompression `
                -ExecutablePath $cpuEngine.Executable `
                -PackagePath $packagePath `
                -Engine $cpuEngine.Name `
                -BuildId $buildId `
                -InferenceCompatibilityId $inferenceCompatibilityId
            $testedCpuEngineCount++
        }
        if ($testedCpuEngineCount -eq 0) {
            throw "No packaged CPU engine could run the local model quality test"
        }
    } finally {
        if (-not $preservedModel) {
            Remove-PackagedModel -PackagePath $packagePath -ModelPath $packagedModelPath
        }
    }
}

if ($Zip) {
    $zipPath = "$packagePath.zip"
    if (Test-Path $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -LiteralPath $packagePath -DestinationPath $zipPath
}

Write-Host "Package created:"
Write-Host $packagePath
if ($Zip) {
    Write-Host "Zip created:"
    Write-Host $zipPath
}
