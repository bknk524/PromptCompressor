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
    $packagePath = Join-Path $projectParent "prompt-compressor-project-exe"
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
$releaseExe = Join-Path $projectRoot "target\release\prompt-compressor-desktop.exe"
$packageExe = Join-Path $packagePath "PromptCompressor.exe"

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
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
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

function Stop-PackageProcesses {
    param([Parameter(Mandatory = $true)][string]$PackagePath)

    $packageFull = [System.IO.Path]::GetFullPath($PackagePath)
    $processes = Get-CimInstance Win32_Process | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_.CommandLine) -and
        $_.CommandLine.IndexOf($packageFull, [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        ($_.Name -eq "PromptCompressor.exe" -or $_.Name -eq "msedgewebview2.exe")
    }

    $desktopProcesses = $processes | Where-Object { $_.Name -eq "PromptCompressor.exe" }
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

function Test-PackagedLocalModelCompression {
    param(
        [Parameter(Mandatory = $true)][string]$PackagePath,
        [Parameter(Mandatory = $true)][string]$SettingsDir
    )

    Write-Host "Running packaged local model compression smoke test..."
    $sampleBase64 = "UmVhY3Qg44Gu5qSc57Si55S76Z2i44Gn44CB5qSc57Si44Oc44K/44Oz44KS5oq844GX44Gf44Go44GN44Gg44GRIEFQSSDjgpLlkbzjgbPlh7rjgZfjgabjgY/jgaDjgZXjgYTjgILml6LlrZjjga4gdXNlU2VhcmNoUGFyYW1zIOOBq+OCiOOCiyBVUkwg44Kv44Ko44Oq566h55CG44Gv57at5oyB44GX44CB5aSn6KaP5qih44Gq44Oq44OV44Kh44Kv44K/44Oq44Oz44Kw44Gv6YG/44GR44Gm44GP44Gg44GV44GE44CC"
    $sample = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($sampleBase64))
    Push-Location $projectRoot
    try {
        Use-LlvmBuildTools
        $jsonLines = & cargo run --release -q -p prompt-compressor-cli -- --settings-dir $SettingsDir --profile internal_llm --format json --level 2 $sample
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged local model smoke test failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $json = ($jsonLines | Out-String).Trim()
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
    if ($result.should_send_original) {
        throw "Packaged local model smoke test returned original prompt: $($result.fallback_reason)"
    }
    if ([string]::IsNullOrWhiteSpace($result.distilled_prompt)) {
        throw "Packaged local model smoke test returned an empty compressed prompt"
    }
    if ($result.metrics.output_characters -ge $result.metrics.input_characters) {
        throw "Packaged local model smoke test did not reduce character count"
    }

    Write-Host "Packaged local model smoke test passed:"
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
    Push-Location $projectRoot
    try {
        Use-LlvmBuildTools
        cargo build --release -p prompt-compressor-desktop
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path $releaseExe)) {
    throw "Release executable not found: $releaseExe"
}

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
Assert-ChildPath -Parent $outputRootPath -Child $packagePath

if (Test-Path $packagePath) {
    if (-not $Clean) {
        throw "Package path already exists. Re-run with -Clean to replace it: $packagePath"
    }
    Stop-PackageProcesses -PackagePath $packagePath
    Remove-Item -LiteralPath $packagePath -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $packagePath | Out-Null
Copy-Item -LiteralPath $releaseExe -Destination $packageExe -Force

Copy-Directory -Source $configPath -Destination (Join-Path $packagePath "application\config")
Copy-Directory -Source $resourcesPath -Destination (Join-Path $packagePath "application\resources")

$profilesPath = Join-Path $configPath "compression-profiles.yaml"
$modelsPath = Join-Path $configPath "model-catalog.yaml"
$runtimesPath = Join-Path $configPath "runtime-backends.yaml"
$defaultProfile = Read-DefaultProfile -Path $profilesPath
$modelRef = Read-ProfileValue -Path $profilesPath -ProfileId $defaultProfile -Key "model_ref"
$runtimeRef = Read-ProfileValue -Path $profilesPath -ProfileId $defaultProfile -Key "runtime_ref"
$modelPath = Read-ProfileValue -Path $modelsPath -ProfileId $modelRef -Key "model_path"
$runtimeExecutablePath = Try-Read-ProfileValue -Path $runtimesPath -ProfileId $runtimeRef -Key "executable_path"

Copy-RelativeFile -RelativePath $modelPath
Assert-PackagedFile -RelativePath "application\$modelPath" -MinimumBytes 1048576

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
Prompt Compressor desktop package

Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Executable: PromptCompressor.exe
Default profile: $defaultProfile
Model: application/$modelPath
Runtime: $runtimeManifest
Desktop transport: WebView2 custom protocol (no HTTP server, no localhost port)

This folder is a runnable package. It is intentionally separate from the source
project folder and contains only the desktop executable plus runtime assets.
"@
$manifest | Set-Content -Encoding UTF8 (Join-Path $packagePath "PACKAGE_MANIFEST.txt")

if (-not $SkipLocalModelSmokeTest) {
    Test-PackagedLocalModelCompression `
        -PackagePath $packagePath `
        -SettingsDir (Join-Path $packagePath "application\config")
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
