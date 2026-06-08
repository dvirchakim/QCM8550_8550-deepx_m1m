# Qualcomm QCS8550 Board - Complete Application Restore Script
# This script restores the DeepX demo application backup to a clean board

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupDir
)

$ADB = "C:\platform-tools\platform-tools\adb.exe"
$DEVICE = "a9ef4ffe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Qualcomm Board Restore Utility" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Validate backup directory
if (-not (Test-Path $BackupDir)) {
    Write-Host "ERROR: Backup directory not found: $BackupDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$BackupDir\BACKUP_MANIFEST.txt")) {
    Write-Host "ERROR: Invalid backup directory (missing manifest)" -ForegroundColor Red
    exit 1
}

Write-Host "Backup Directory: $BackupDir" -ForegroundColor White
Write-Host ""
Write-Host "WARNING: This will restore the DeepX demo to the board!" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to cancel, or any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Get kernel version from backup
$KERNEL_VER = (Get-Content "$BackupDir\system_info\uname.txt" | Select-String -Pattern "Linux.*(\d+\.\d+\.\d+-\S+)" | ForEach-Object { $_.Matches.Groups[1].Value })
if (-not $KERNEL_VER) {
    Write-Host "WARNING: Could not detect kernel version from backup, using current board kernel" -ForegroundColor Yellow
    $KERNEL_VER = (& cmd /c "$ADB -s $DEVICE shell uname -r").Trim()
}
Write-Host "Target Kernel Version: $KERNEL_VER" -ForegroundColor Cyan
Write-Host ""

# Stop existing service if running
Write-Host "[1/12] Stopping existing demo service..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell systemctl stop imdt-deepx-demo 2>&1" | Out-Null
& cmd /c "$ADB -s $DEVICE shell systemctl disable imdt-deepx-demo 2>&1" | Out-Null
Start-Sleep -Seconds 2
Write-Host "   Service stopped" -ForegroundColor Green

# Make filesystem writable
Write-Host "[2/12] Remounting filesystem as read-write..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell mount -o rw,remount /"
Write-Host "   Filesystem remounted" -ForegroundColor Green

# Restore DeepX application
Write-Host "[3/12] Restoring DeepX application..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell rm -rf /usr/share/dx-stream"
& cmd /c "$ADB -s $DEVICE push $BackupDir\usr_share_dx-stream /usr/share/dx-stream 2>&1"
& cmd /c "$ADB -s $DEVICE shell chmod +x /usr/share/dx-stream/imdt-deepx-demo.sh"
Write-Host "   DeepX application restored" -ForegroundColor Green

# Restore AI models
Write-Host "[4/12] Restoring AI models..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell mkdir -p /opt"
& cmd /c "$ADB -s $DEVICE push $BackupDir\opt /opt 2>&1"
& cmd /c "$ADB -s $DEVICE shell chmod +x /opt/*.tflite"
& cmd /c "$ADB -s $DEVICE shell chmod +x /opt/*.labels"
& cmd /c "$ADB -s $DEVICE shell chmod +x /opt/*.sh"
Write-Host "   AI models restored" -ForegroundColor Green

# Restore systemd service
Write-Host "[5/12] Restoring systemd service..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE push $BackupDir\systemd\imdt-deepx-demo.service /lib/systemd/system/imdt-deepx-demo.service 2>&1"
& cmd /c "$ADB -s $DEVICE shell chmod 644 /lib/systemd/system/imdt-deepx-demo.service"
Write-Host "   Systemd service restored" -ForegroundColor Green

# Restore kernel modules
Write-Host "[6/12] Restoring kernel modules..." -ForegroundColor Yellow
if (Test-Path "$BackupDir\kernel_modules\extra") {
    & cmd /c "$ADB -s $DEVICE shell mkdir -p /lib/modules/$KERNEL_VER/extra"
    & cmd /c "$ADB -s $DEVICE push $BackupDir\kernel_modules\extra /lib/modules/$KERNEL_VER/extra 2>&1"
    & cmd /c "$ADB -s $DEVICE shell chmod 644 /lib/modules/$KERNEL_VER/extra/*.ko"
    Write-Host "   Kernel modules restored" -ForegroundColor Green
} else {
    Write-Host "   No kernel modules in backup (skipped)" -ForegroundColor Yellow
}

# Restore GStreamer plugins
Write-Host "[7/12] Restoring GStreamer plugins..." -ForegroundColor Yellow
if (Test-Path "$BackupDir\lib_gstreamer\gstreamer-1.0") {
    & cmd /c "$ADB -s $DEVICE shell mkdir -p /usr/lib/gstreamer-1.0"
    & cmd /c "$ADB -s $DEVICE push $BackupDir\lib_gstreamer\gstreamer-1.0 /usr/lib/gstreamer-1.0 2>&1"
    Write-Host "   GStreamer plugins restored" -ForegroundColor Green
} else {
    Write-Host "   No GStreamer plugins in backup (skipped)" -ForegroundColor Yellow
}

# Load kernel module
Write-Host "[8/12] Loading DeepX kernel module..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell modprobe dxrt_driver 2>&1" | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   Kernel module loaded" -ForegroundColor Green
} else {
    Write-Host "   WARNING: Could not load kernel module (may need manual intervention)" -ForegroundColor Yellow
}

# Reload systemd daemon
Write-Host "[9/12] Reloading systemd daemon..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell systemctl daemon-reload"
Write-Host "   Systemd daemon reloaded" -ForegroundColor Green

# Enable service
Write-Host "[10/12] Enabling demo service..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell systemctl enable imdt-deepx-demo"
Write-Host "   Service enabled for auto-start" -ForegroundColor Green

# Verify installation
Write-Host "[11/12] Verifying installation..." -ForegroundColor Yellow
$VERIFY_OK = $true

if (-not (& cmd /c "$ADB -s $DEVICE shell test -f /usr/share/dx-stream/imdt-deepx-demo.sh && echo OK")) {
    Write-Host "   ERROR: Demo script not found!" -ForegroundColor Red
    $VERIFY_OK = $false
}

if (-not (& cmd /c "$ADB -s $DEVICE shell test -f /lib/systemd/system/imdt-deepx-demo.service && echo OK")) {
    Write-Host "   ERROR: Systemd service not found!" -ForegroundColor Red
    $VERIFY_OK = $false
}

if (-not (& cmd /c "$ADB -s $DEVICE shell test -f /opt/YOLOv8-Detection-Quantized.tflite && echo OK")) {
    Write-Host "   ERROR: AI models not found!" -ForegroundColor Red
    $VERIFY_OK = $false
}

if ($VERIFY_OK) {
    Write-Host "   Verification passed" -ForegroundColor Green
} else {
    Write-Host "   Verification FAILED - check errors above" -ForegroundColor Red
}

# Start service
Write-Host "[12/12] Starting demo service..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell systemctl start imdt-deepx-demo"
Start-Sleep -Seconds 3

$SERVICE_STATUS = & cmd /c "$ADB -s $DEVICE shell systemctl is-active imdt-deepx-demo"
if ($SERVICE_STATUS -match "active") {
    Write-Host "   Service started successfully" -ForegroundColor Green
} else {
    Write-Host "   WARNING: Service may not have started properly" -ForegroundColor Yellow
    Write-Host "   Check logs with: adb shell journalctl -u imdt-deepx-demo -f" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RESTORE COMPLETED!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Status Check:" -ForegroundColor Yellow
Write-Host "  Service Status: $SERVICE_STATUS" -ForegroundColor White
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  Check service status: adb shell systemctl status imdt-deepx-demo" -ForegroundColor White
Write-Host "  View logs: adb shell journalctl -u imdt-deepx-demo -f" -ForegroundColor White
Write-Host "  Stop service: adb shell systemctl stop imdt-deepx-demo" -ForegroundColor White
Write-Host "  Start service: adb shell systemctl start imdt-deepx-demo" -ForegroundColor White
Write-Host "  Reboot board: adb reboot" -ForegroundColor White
Write-Host ""
Write-Host "The demo should now be running and will auto-start on boot!" -ForegroundColor Green
Write-Host ""
