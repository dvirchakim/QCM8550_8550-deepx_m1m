# =============================================================================
# QCS8550 Demo Picker Backup Script
# Backs up the demo picker app + bundled demos to GitHub repo
# Repo: https://github.com/dvirchakim/QCM8550_8550-deepx_m1m
# =============================================================================

$ErrorActionPreference = "Stop"
$GITHUB_REPO = "https://github.com/dvirchakim/QCM8550_8550-deepx_m1m.git"
$BACKUP_SUBDIR = "demo_picker_backup"

# ---------------------------------------------------------------------------
# 1. Find ADB
# ---------------------------------------------------------------------------
Write-Host "`n[1/6] Locating ADB..." -ForegroundColor Cyan

$adb = $null

# Helper: test if a path is a real, working ADB binary
function Test-AdbWorks($path) {
    if (-not (Test-Path $path)) { return $false }
    try {
        $out = & $path version 2>&1
        return ($out -match "Android Debug Bridge")
    } catch { return $false }
}

# Collect all candidates (PATH first, then common locations)
$allCandidates = @()
try {
    $fromPath = (Get-Command adb -ErrorAction Stop).Source
    $allCandidates += $fromPath
} catch {}

$allCandidates += @(
    "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
    "$env:ProgramFiles\Android\platform-tools\adb.exe",
    "$env:ProgramFiles(x86)\Android\platform-tools\adb.exe",
    "C:\Android\platform-tools\adb.exe",
    "C:\adb\adb.exe",
    "C:\platform-tools\adb.exe",
    "$env:USERPROFILE\AppData\Local\Android\Sdk\platform-tools\adb.exe",
    "$env:USERPROFILE\platform-tools\adb.exe",
    "$env:USERPROFILE\Downloads\platform-tools\adb.exe"
)

foreach ($c in $allCandidates) {
    if (Test-AdbWorks $c) {
        $adb = $c
        Write-Host "  Found working ADB: $adb" -ForegroundColor Green
        & $adb version 2>&1 | Select-String "Android Debug Bridge" | ForEach-Object { Write-Host "  $_" }
        break
    } elseif (Test-Path $c) {
        Write-Host "  Skipping non-functional ADB stub: $c" -ForegroundColor Yellow
    }
}

if (-not $adb) {
    Write-Host "`n  ADB not found or not functional. Attempting auto-download of platform-tools..." -ForegroundColor Yellow
    $zipUrl  = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"
    $zipDest = "$env:TEMP\platform-tools.zip"
    $ptDir   = "C:\platform-tools"
    try {
        Write-Host "  Downloading $zipUrl ..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipDest -UseBasicParsing
        Write-Host "  Extracting to $ptDir ..."
        Expand-Archive -Path $zipDest -DestinationPath "C:\" -Force
        $adb = "$ptDir\adb.exe"
        if (Test-AdbWorks $adb) {
            Write-Host "  platform-tools installed at $ptDir" -ForegroundColor Green
        } else {
            throw "Extracted ADB doesn't work"
        }
    } catch {
        Write-Host "`n  ERROR: Could not auto-install ADB. Please:" -ForegroundColor Red
        Write-Host "    1. Download: https://developer.android.com/studio/releases/platform-tools"
        Write-Host "    2. Extract to C:\platform-tools\"
        Write-Host "    3. Re-run this script"
        exit 1
    }
}

# ---------------------------------------------------------------------------
# 2. Check device connectivity
# ---------------------------------------------------------------------------
Write-Host "`n[2/6] Checking ADB device..." -ForegroundColor Cyan

$devices = & $adb devices 2>&1
Write-Host $devices

$deviceLine = ($devices | Select-String -Pattern "^\S+\s+(device|unauthorized|offline)" | Select-Object -First 1)
if (-not $deviceLine) {
    Write-Host "`n  ERROR: No device found. Make sure:" -ForegroundColor Red
    Write-Host "    - The QCS8550 is powered on and connected via USB"
    Write-Host "    - USB debugging is enabled on the device"
    Write-Host "    - You have accepted the RSA key prompt on the device"
    exit 1
}

if ($deviceLine -match "unauthorized") {
    Write-Host "`n  ERROR: Device is unauthorized. Accept the debug prompt on the QCS8550 screen." -ForegroundColor Red
    exit 1
}

$deviceId = ($deviceLine.Matches[0].Value -split "\s+")[0]
Write-Host "  Device connected: $deviceId" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Gain root access
# ---------------------------------------------------------------------------
Write-Host "`n[3/6] Gaining root via ADB..." -ForegroundColor Cyan

$rootResult = & $adb -s $deviceId root 2>&1
Write-Host "  $rootResult"

Start-Sleep -Seconds 2

# Verify root
$whoami = & $adb -s $deviceId shell whoami 2>&1
Write-Host "  Shell user: $whoami"
if ($whoami -notmatch "root") {
    Write-Host "  WARNING: Not running as root. APK pull may still work, but /data paths may be inaccessible." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 4. Find the demo picker package
# ---------------------------------------------------------------------------
Write-Host "`n[4/6] Finding demo picker application..." -ForegroundColor Cyan

# Search for common demo picker package names
$searchTerms = @("demo", "picker", "launcher", "showcase", "sample")
$allPackages = & $adb -s $deviceId shell pm list packages 2>&1

Write-Host "  All packages (filtering for demo/picker/showcase):"
$matchedPackages = @()
foreach ($pkg in $allPackages) {
    $pkgName = $pkg -replace "^package:", ""
    foreach ($term in $searchTerms) {
        if ($pkgName -like "*$term*") {
            Write-Host "    -> $pkgName" -ForegroundColor Yellow
            $matchedPackages += $pkgName
            break
        }
    }
}

if ($matchedPackages.Count -eq 0) {
    Write-Host "`n  No demo/picker packages found by keyword. Listing ALL 3rd-party packages:" -ForegroundColor Yellow
    $allPackages | Where-Object { $_ -notmatch "com\.android|com\.google|com\.qualcomm\.qti\.qmmi|android" } | ForEach-Object {
        Write-Host "    $_"
    }
    $pkgInput = Read-Host "`n  Enter the exact package name of the demo picker app"
    $matchedPackages = @($pkgInput.Trim())
}

# Deduplicate
$matchedPackages = $matchedPackages | Sort-Object -Unique

# If multiple found, let user choose
if ($matchedPackages.Count -gt 1) {
    Write-Host "`n  Multiple candidates found:" -ForegroundColor Yellow
    for ($i = 0; $i -lt $matchedPackages.Count; $i++) {
        Write-Host "    [$i] $($matchedPackages[$i])"
    }
    $choice = Read-Host "  Enter number of the demo picker package (or press Enter to back up ALL)"
    if ($choice -match "^\d+$" -and [int]$choice -lt $matchedPackages.Count) {
        $matchedPackages = @($matchedPackages[[int]$choice])
    }
}

Write-Host "`n  Will back up: $($matchedPackages -join ', ')" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 5. Pull APKs and data
# ---------------------------------------------------------------------------
Write-Host "`n[5/6] Pulling APKs and demo assets..." -ForegroundColor Cyan

# Set up local backup folder
$backupRoot = Join-Path $PSScriptRoot $BACKUP_SUBDIR
if (-not (Test-Path $backupRoot)) { New-Item -ItemType Directory -Path $backupRoot | Out-Null }

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$backupDir = Join-Path $backupRoot $timestamp
New-Item -ItemType Directory -Path $backupDir | Out-Null
Write-Host "  Backup directory: $backupDir"

foreach ($pkg in $matchedPackages) {
    $pkgDir = Join-Path $backupDir $pkg
    New-Item -ItemType Directory -Path $pkgDir | Out-Null

    # Get APK path(s)
    Write-Host "`n  Package: $pkg" -ForegroundColor Cyan
    $apkPaths = & $adb -s $deviceId shell pm path $pkg 2>&1
    Write-Host "  APK path(s): $apkPaths"

    foreach ($apkLine in $apkPaths) {
        $apkPath = ($apkLine -replace "^package:", "").Trim()
        if ($apkPath -and (Test-Path -IsValid ($apkPath -replace "/", "\"))) {
            $apkFile = Split-Path $apkPath -Leaf
            Write-Host "  Pulling APK: $apkPath -> $pkgDir\$apkFile"
            & $adb -s $deviceId pull $apkPath "$pkgDir\$apkFile" 2>&1 | Write-Host
        }
    }

    # Try to pull app data (requires root)
    Write-Host "  Attempting to pull /data/data/$pkg (requires root)..."
    $dataDir = Join-Path $pkgDir "data"
    New-Item -ItemType Directory -Path $dataDir | Out-Null
    & $adb -s $deviceId pull "/data/data/$pkg" $dataDir 2>&1 | Write-Host

    # Pull OBB / external storage if exists
    $obbDirs = @(
        "/sdcard/Android/obb/$pkg",
        "/sdcard/Android/data/$pkg",
        "/storage/emulated/0/Android/obb/$pkg",
        "/storage/emulated/0/Android/data/$pkg"
    )
    foreach ($obbDir in $obbDirs) {
        $checkResult = & $adb -s $deviceId shell "ls $obbDir 2>/dev/null && echo EXISTS || echo MISSING" 2>&1
        if ($checkResult -match "EXISTS") {
            Write-Host "  Pulling external data: $obbDir"
            $obbLocal = Join-Path $pkgDir "obb_data"
            New-Item -ItemType Directory -Path $obbLocal -Force | Out-Null
            & $adb -s $deviceId pull $obbDir $obbLocal 2>&1 | Write-Host
        }
    }

    # Look for demos in common locations
    Write-Host "  Searching for demo assets in /data/local/tmp, /vendor/app, /system/app..."
    $demoSearchPaths = @("/vendor/app", "/system/app", "/odm/app", "/product/app", "/data/local/tmp")
    foreach ($searchPath in $demoSearchPaths) {
        $found = & $adb -s $deviceId shell "find $searchPath -name '*demo*' -o -name '*Demo*' 2>/dev/null" 2>&1
        if ($found -and $found -notmatch "^$") {
            Write-Host "  Found in ${searchPath}:" -ForegroundColor Yellow
            $found | ForEach-Object { Write-Host "    $_" }

            $assetsDir = Join-Path $pkgDir "vendor_assets"
            New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null
            foreach ($f in $found) {
                if ($f.Trim()) {
                    & $adb -s $deviceId pull $f.Trim() $assetsDir 2>&1 | Write-Host
                }
            }
        }
    }
}

# Write a README with device info
$deviceInfo = & $adb -s $deviceId shell getprop 2>&1
$readmePath = Join-Path $backupDir "DEVICE_INFO.txt"
"Backup Timestamp: $timestamp`n" | Out-File $readmePath
"Device ID: $deviceId`n" | Add-Content $readmePath
"Packages backed up: $($matchedPackages -join ', ')`n`n" | Add-Content $readmePath
"--- Full Device Properties ---`n" | Add-Content $readmePath
$deviceInfo | Add-Content $readmePath

Write-Host "`n  Backup complete at: $backupDir" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 6. Commit and push to GitHub
# ---------------------------------------------------------------------------
Write-Host "`n[6/6] Committing to GitHub..." -ForegroundColor Cyan

# Check git
try { $gitPath = (Get-Command git -ErrorAction Stop).Source }
catch {
    Write-Host "  ERROR: Git not found in PATH. Please install Git from https://git-scm.com" -ForegroundColor Red
    exit 1
}

# Clone or pull the repo into a temp folder
$repoDir = Join-Path $env:TEMP "QCM8550_deepx_repo_$(Get-Random)"
Write-Host "  Cloning repo to: $repoDir"
& git clone $GITHUB_REPO $repoDir 2>&1 | Write-Host

# Copy backup into repo
$repoBackupDir = Join-Path $repoDir $BACKUP_SUBDIR
if (-not (Test-Path $repoBackupDir)) { New-Item -ItemType Directory -Path $repoBackupDir | Out-Null }

$destDir = Join-Path $repoBackupDir $timestamp
Copy-Item -Path $backupDir -Destination $destDir -Recurse -Force
Write-Host "  Copied backup to repo: $destDir" -ForegroundColor Green

# Create a .gitattributes to handle APK binary files
$gitattr = Join-Path $repoDir ".gitattributes"
if (-not (Test-Path $gitattr)) {
    "*.apk binary`n*.obb binary`n*.so binary" | Out-File $gitattr -Encoding utf8
    Write-Host "  Created .gitattributes for binary files"
}

# Git operations
Push-Location $repoDir
try {
    & git add -A 2>&1 | Write-Host
    & git commit -m "backup: demo picker app + demos from QCS8550 [$timestamp]" 2>&1 | Write-Host
    Write-Host "`n  Pushing to GitHub (you may be prompted for credentials)..." -ForegroundColor Cyan
    & git push origin main 2>&1 | Write-Host
    if ($LASTEXITCODE -ne 0) {
        # Try 'master' branch if 'main' fails
        & git push origin master 2>&1 | Write-Host
    }
    Write-Host "`n  Successfully pushed to GitHub!" -ForegroundColor Green
}
finally {
    Pop-Location
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " DONE! Demo picker backed up and pushed to GitHub." -ForegroundColor Green
Write-Host " Repo: $GITHUB_REPO" -ForegroundColor Green
Write-Host " Local backup: $backupDir" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green
