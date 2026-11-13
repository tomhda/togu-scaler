<#
.SYNOPSIS
    TOGU SCALER のセットアップ（togu-scaler-setup-win64.exe）を作る。

.DESCRIPTION
    1. scripts\build_portable.ps1 -WithHelper -NoZip で portable_dist\togu-scaler を作る
       （-SkipPortableBuild なら既存のフォルダを使う）。
    2. Inno Setup 7.1.0 を公式リリースから取得し、SHA-256 と署名を確かめる
       （scripts\.cache に置き、2 回目からは使い回す）。
    3. installer\togu-scaler.iss をコンパイルし、出力先に書き出す。

.EXAMPLE
    pwsh -File scripts\build_installer.ps1
    pwsh -File scripts\build_installer.ps1 -SkipPortableBuild -OutputDirectory tmp\installer
#>
param(
    [string]$OutputDirectory = "",
    [switch]$SkipPortableBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$app = Join-Path $repo "portable_dist\togu-scaler"
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo "installer\output" }
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not $outputPath.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "出力先はリポジトリの中を指定してください: $outputPath"
}

if (-not $SkipPortableBuild) {
    & (Join-Path $PSScriptRoot "build_portable.ps1") -WithHelper -NoZip
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "build_portable.ps1 が失敗しました (exit=$LASTEXITCODE)"
    }
}

# 中身の点検。GPU 用モデルと同梱ツールが揃っていること、NPU・AdcSR・FILM の
# キットの中身が混ざっていないこと、上流のサンプル（メディア）が入っていないこと。
$required = @(
    "togu-scaler.exe", "togu-scaler-cli.exe", "_internal",
    "vendor\realesrgan\realesrgan-ncnn-vulkan.exe", "vendor\rife\rife-ncnn-vulkan.exe",
    "vendor\ffmpeg\bin\ffmpeg.exe", "vendor\ffmpeg\bin\ffprobe.exe",
    "vendor\ffmpeg\LICENSE", "vendor\winml-sr\winml-sr.exe",
    "models\custom\README.txt", "LICENSE.txt",
    "THIRD-PARTY-NOTICES.txt", "THIRD-PARTY-NOTICES.en.txt",
    "licenses\PySide6\LGPL-3.0.txt", "licenses\PySide6\GPL-3.0.txt",
    "はじめに.txt", "Getting started.txt"
)
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $app $relative))) {
        throw "配布フォルダに必要なファイルがありません: $relative"
    }
}
$forbidden = @("tools", "vendor\winml-film", "models\film", "vendor\amd-npu-1.8")
foreach ($relative in $forbidden) {
    if (Test-Path -LiteralPath (Join-Path $app $relative)) {
        throw "配布フォルダに本体へ入れないものがあります: $relative"
    }
}
$kitModels = @(Get-ChildItem -LiteralPath (Join-Path $app "models\ai") -File | Where-Object {
    $_.Name -like "*bf16cast*" -or $_.Name -like "*_body_*" -or $_.Name -like "adcsr*"
})
if ($kitModels.Count -gt 0) {
    throw "配布フォルダにキット用のモデルがあります: $($kitModels.Name -join ', ')"
}
$internal = Join-Path $app "_internal"
$media = @(Get-ChildItem -LiteralPath $app -Recurse -File | Where-Object {
    $_.Extension -in @(".jpg", ".jpeg", ".png", ".mp4", ".mkv", ".mov", ".webm", ".gif") -and
    -not $_.FullName.StartsWith($internal + "\", [System.StringComparison]::OrdinalIgnoreCase)
})
if ($media.Count -gt 0) {
    throw "配布フォルダにメディアファイルがあります: $($media.FullName -join ', ')"
}

$versionLine = (& (Join-Path $app "togu-scaler-cli.exe") --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $versionLine -notmatch '^togu-scaler (\d+\.\d+\.\d+)$') {
    throw "togu-scaler-cli.exe から版を取得できません: $versionLine"
}
$appVersion = $Matches[1]

# --- Inno Setup 7.1.0（ポータブル展開して使う） ---
$compilerVersion = "7.1.0"
$compilerCache = Join-Path $PSScriptRoot (".cache\innosetup-" + $compilerVersion)
$compilerHome = Join-Path $compilerCache "portable"
$compilerPath = Join-Path $compilerHome "ISCC.exe"
$bootstrap = Join-Path $compilerCache ("innosetup-" + $compilerVersion + "-x64.exe")
$expectedBootstrapHash = "0362a383ed217d4c4239b5933866dd96d3eb2102737da92f80f6057a4b40df2f"
$expectedCompilerHash = "d06ebd38f38e3cee60a3c50cc45bd449d77e0bc6a5cabc607ea9886808e4de1a"
$downloadUrl = "https://github.com/jrsoftware/issrc/releases/download/is-7_1_0/innosetup-7.1.0-x64.exe"

function Assert-PyrsysSignature([string]$Path) {
    $signature = Get-AuthenticodeSignature -FilePath $Path
    if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid -or
        -not $signature.SignerCertificate -or
        $signature.SignerCertificate.GetNameInfo(
            [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) -ne "Pyrsys B.V.") {
        throw "Inno Setup の署名を確認できません: $Path ($($signature.StatusMessage))"
    }
}

New-Item -ItemType Directory -Path $compilerCache -Force | Out-Null
if (-not (Test-Path -LiteralPath $compilerPath -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
        Write-Output "Inno Setup $compilerVersion を公式リリースから取得します。"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $downloadUrl -OutFile $bootstrap -UseBasicParsing
    }
    $hash = (Get-FileHash -LiteralPath $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $expectedBootstrapHash) {
        throw "Inno Setup のインストーラーの SHA-256 が一致しません: $bootstrap"
    }
    Assert-PyrsysSignature $bootstrap
    New-Item -ItemType Directory -Path $compilerHome -Force | Out-Null
    $arguments = '/PORTABLE=1 /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="' + $compilerHome + '"'
    $process = Start-Process -FilePath $bootstrap -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $compilerPath -PathType Leaf)) {
        throw "Inno Setup の展開に失敗しました (exit=$($process.ExitCode))"
    }
}
$compilerHash = (Get-FileHash -LiteralPath $compilerPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($compilerHash -ne $expectedCompilerHash) {
    throw "ISCC.exe の SHA-256 が一致しません: $compilerPath"
}
Assert-PyrsysSignature $compilerPath
$compilerOutput = (& $compilerPath --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $compilerOutput -ne $compilerVersion) {
    throw "Inno Setup の版が想定と違います: $compilerOutput"
}

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$setupPath = Join-Path $outputPath "togu-scaler-setup-win64.exe"
if (Test-Path -LiteralPath $setupPath) { Remove-Item -LiteralPath $setupPath -Force }

$saved = @{}
foreach ($name in @("TOGU_APP_VERSION", "TOGU_PUBLISH_DIR", "TOGU_REPO_DIR")) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
    $env:TOGU_APP_VERSION = $appVersion
    $env:TOGU_PUBLISH_DIR = $app
    $env:TOGU_REPO_DIR = $repo
    Write-Output "TOGU SCALER のセットアップを作ります (version $appVersion)。"
    & $compilerPath --quiet-progress ("--output-dir=$outputPath") (Join-Path $repo "installer\togu-scaler.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup のコンパイルが失敗しました (exit=$LASTEXITCODE)" }
}
finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process")
    }
}

if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
    throw "セットアップが出力されていません: $setupPath"
}
$setup = Get-Item -LiteralPath $setupPath
Write-Output ("SETUP={0}" -f $setup.FullName)
Write-Output ("SIZE_MB={0:N1}" -f ($setup.Length / 1MB))
Write-Output ("SHA256={0}" -f (Get-FileHash -LiteralPath $setupPath -Algorithm SHA256).Hash.ToLowerInvariant())
