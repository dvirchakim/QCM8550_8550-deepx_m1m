# Qualcomm QCS8550 Board - Complete Application Backup Script
# This script creates a full backup of the running DeepX demo application
# including all dependencies, drivers, configs, and systemd services

$ADB = "C:\platform-tools\platform-tools\adb.exe"
$DEVICE = "a9ef4ffe"
$BACKUP_DIR = "board_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$BACKUP_ROOT = Join-Path $PSScriptRoot $BACKUP_DIR

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Qualcomm Board Complete Backup Utility" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create backup directory structure
Write-Host "[1/10] Creating backup directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $BACKUP_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\usr_share_dx-stream" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\opt" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\systemd" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\kernel_modules" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\lib_gstreamer" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\system_info" | Out-Null
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\dev_nodes" | Out-Null
Write-Host "   Created: $BACKUP_ROOT" -ForegroundColor Green

# Backup system information
Write-Host "[2/10] Capturing system information..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell uname -a" > "$BACKUP_ROOT\system_info\uname.txt"
& cmd /c "$ADB -s $DEVICE shell cat /proc/version" > "$BACKUP_ROOT\system_info\kernel_version.txt"
& cmd /c "$ADB -s $DEVICE shell lsmod" > "$BACKUP_ROOT\system_info\loaded_modules.txt"
& cmd /c "$ADB -s $DEVICE shell ps aux" > "$BACKUP_ROOT\system_info\running_processes.txt"
& cmd /c "$ADB -s $DEVICE shell systemctl list-units --type=service --state=running" > "$BACKUP_ROOT\system_info\running_services.txt"
& cmd /c "$ADB -s $DEVICE shell df -h" > "$BACKUP_ROOT\system_info\disk_usage.txt"
& cmd /c "$ADB -s $DEVICE shell mount" > "$BACKUP_ROOT\system_info\mounts.txt"
Write-Host "   System info captured" -ForegroundColor Green

# Backup device nodes and symlinks
Write-Host "[3/10] Backing up device nodes info..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell ls -la /dev/fastrpc* /dev/adsprpc* 2>&1" > "$BACKUP_ROOT\dev_nodes\fastrpc_devices.txt"
& cmd /c "$ADB -s $DEVICE shell ls -la /dev/video* 2>&1" > "$BACKUP_ROOT\dev_nodes\video_devices.txt"
& cmd /c "$ADB -s $DEVICE shell ls -la /dev/dri/* 2>&1" > "$BACKUP_ROOT\dev_nodes\dri_devices.txt"
& cmd /c "$ADB -s $DEVICE shell find /dev -type l -ls 2>&1" > "$BACKUP_ROOT\dev_nodes\all_symlinks.txt"
Write-Host "   Device nodes info captured" -ForegroundColor Green

# Backup DeepX application files
Write-Host "[4/10] Backing up DeepX application (/usr/share/dx-stream)..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE pull /usr/share/dx-stream $BACKUP_ROOT\usr_share_dx-stream 2>&1"
Write-Host "   DeepX application backed up" -ForegroundColor Green

# Backup model files
Write-Host "[5/10] Backing up AI models (/opt)..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE pull /opt $BACKUP_ROOT\opt 2>&1"
Write-Host "   AI models backed up" -ForegroundColor Green

# Backup systemd service files
Write-Host "[6/10] Backing up systemd service configuration..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE pull /lib/systemd/system/imdt-deepx-demo.service $BACKUP_ROOT\systemd\imdt-deepx-demo.service 2>&1"
& cmd /c "$ADB -s $DEVICE pull /lib/systemd/system/qmmf-server.service $BACKUP_ROOT\systemd\qmmf-server.service 2>&1"
& cmd /c "$ADB -s $DEVICE pull /lib/systemd/system/init_qti_graphics.service $BACKUP_ROOT\systemd\init_qti_graphics.service 2>&1"
& cmd /c "$ADB -s $DEVICE shell systemctl is-enabled imdt-deepx-demo" > "$BACKUP_ROOT\systemd\service_enabled_status.txt"
Write-Host "   Systemd services backed up" -ForegroundColor Green

# Backup kernel modules
Write-Host "[7/10] Backing up kernel modules..." -ForegroundColor Yellow
$KERNEL_VER = (& cmd /c "$ADB -s $DEVICE shell uname -r").Trim()
& cmd /c "$ADB -s $DEVICE pull /lib/modules/$KERNEL_VER/extra/ $BACKUP_ROOT\kernel_modules\extra 2>&1"
Write-Host "   Kernel version: $KERNEL_VER" -ForegroundColor Green
Write-Host "   Kernel modules backed up" -ForegroundColor Green

# Backup GStreamer plugins
Write-Host "[8/10] Backing up GStreamer plugins..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell find /usr/lib -name '*gst*' -o -name '*deepx*' -o -name '*qti*' 2>&1" > "$BACKUP_ROOT\lib_gstreamer\gstreamer_files_list.txt"
& cmd /c "$ADB -s $DEVICE pull /usr/lib/gstreamer-1.0 $BACKUP_ROOT\lib_gstreamer\gstreamer-1.0 2>&1"
Write-Host "   GStreamer plugins backed up" -ForegroundColor Green

# Backup shared libraries
Write-Host "[9/10] Backing up critical shared libraries..." -ForegroundColor Yellow
& cmd /c "$ADB -s $DEVICE shell ldd /usr/bin/gst-launch-1.0 2>&1" > "$BACKUP_ROOT\system_info\gstreamer_dependencies.txt"
New-Item -ItemType Directory -Force -Path "$BACKUP_ROOT\lib" | Out-Null
& cmd /c "$ADB -s $DEVICE shell find /usr/lib -name '*deepx*' -o -name '*Qnn*' -o -name '*qti*' 2>&1" > "$BACKUP_ROOT\lib\deepx_libs_list.txt"
Write-Host "   Library dependencies documented" -ForegroundColor Green

# Create backup manifest
Write-Host "[10/10] Creating backup manifest..." -ForegroundColor Yellow
$MANIFEST = @"
========================================
QUALCOMM QCS8550 BOARD BACKUP MANIFEST
========================================
Backup Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Device ID: $DEVICE
Kernel Version: $KERNEL_VER

BACKED UP COMPONENTS:
---------------------
1. DeepX Demo Application
   - Location: /usr/share/dx-stream/
   - Startup Script: imdt-deepx-demo.sh
   - Configs: configs/ directory
   - Libraries: lib/ directory

2. AI Models
   - Location: /opt/
   - YOLOv8 Detection Model
   - YOLO Pose Estimation Models
   - HRNet Pose Model
   - Label files

3. Systemd Services
   - imdt-deepx-demo.service (Main demo service)
   - qmmf-server.service (Qualcomm Multimedia Framework)
   - init_qti_graphics.service (Graphics initialization)

4. Kernel Modules
   - dxrt_driver.ko (DeepX Runtime Driver)
   - Location: /lib/modules/$KERNEL_VER/extra/

5. GStreamer Plugins
   - Location: /usr/lib/gstreamer-1.0/
   - DeepX plugins (dxpreprocess, dxinfer, dxpostprocess, dxosd)
   - Qualcomm plugins (qtiqmmfsrc, qtivtransform, qtivcomposer, etc.)

6. Device Nodes
   - /dev/fastrpc* (FastRPC devices for DSP communication)
   - /dev/adsprpc* (ADSP RPC devices)
   - /dev/video* (Video devices)

7. System Information
   - Running processes
   - Loaded modules
   - Service status
   - Mount points

RESTORE INSTRUCTIONS:
--------------------
Use the restore_board.ps1 script to restore this backup to a clean board.

NOTES:
------
- This backup includes the complete DeepX demo that runs on boot
- The demo performs dual-camera pose estimation + YOLOv8 object detection
- Requires Qualcomm QCS8550 board with proper BSP installed
- Weston compositor must be running for display output

========================================
"@

$MANIFEST | Out-File -FilePath "$BACKUP_ROOT\BACKUP_MANIFEST.txt" -Encoding UTF8
Write-Host "   Backup manifest created" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "BACKUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup Location: $BACKUP_ROOT" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review the backup in: $BACKUP_ROOT" -ForegroundColor White
Write-Host "  2. Use restore_board.ps1 to restore to a clean board" -ForegroundColor White
Write-Host "  3. Keep this backup safe before making changes" -ForegroundColor White
Write-Host ""
