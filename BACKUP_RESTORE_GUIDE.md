# Qualcomm QCS8550 Board - Backup & Restore Guide

## Overview
This guide explains how to backup and restore the complete DeepX demo application running on your Qualcomm QCS8550 board.

## Current Running Application

**Application**: IMDT Qualcomm + DeepX Demo  
**Description**: Dual-camera pose estimation with YOLOv8 object detection  
**Auto-start**: Yes (systemd service enabled on boot)  
**Service Name**: `imdt-deepx-demo.service`

### What's Running
- **Cameras**: 2x cameras (camera0 and camera1)
- **AI Models**:
  - YOLOv5 Pose Estimation (640x640) on both cameras
  - YOLOv8 Object Detection (quantized for HTP)
- **Display**: Wayland compositor with 5-panel grid layout
- **Hardware Acceleration**: Qualcomm HTP (Hexagon Tensor Processor)

### Key Components
1. **Application Files**: `/usr/share/dx-stream/`
2. **AI Models**: `/opt/` (YOLOv8, YOLOv5, HRNet models)
3. **Systemd Service**: `/lib/systemd/system/imdt-deepx-demo.service`
4. **Kernel Module**: `dxrt_driver.ko` (DeepX Runtime Driver)
5. **GStreamer Plugins**: DeepX and Qualcomm plugins in `/usr/lib/gstreamer-1.0/`

---

## Backup Instructions

### Step 1: Run Backup Script
```powershell
cd c:\Users\dvir\CascadeProjects\qualcomm+deepx_m1
.\backup_board.ps1
```

### Step 2: Wait for Completion
The script will:
- Create a timestamped backup directory (e.g., `board_backup_20260525_173000`)
- Pull all application files, models, configs, and system info
- Generate a detailed manifest

### Step 3: Verify Backup
Check the backup directory for:
- `BACKUP_MANIFEST.txt` - Complete backup details
- `usr_share_dx-stream/` - Application files
- `opt/` - AI models
- `systemd/` - Service configurations
- `kernel_modules/` - Kernel drivers
- `system_info/` - System state snapshots

### Backup Size
Expect approximately **100-200 MB** depending on the number of models.

---

## Restore Instructions

### Prerequisites
- Clean Qualcomm QCS8550 board with BSP installed
- ADB connection established
- Board should have basic system services running

### Step 1: Run Restore Script
```powershell
cd c:\Users\dvir\CascadeProjects\qualcomm+deepx_m1
.\restore_board.ps1 -BackupDir ".\board_backup_YYYYMMDD_HHMMSS"
```

Replace `YYYYMMDD_HHMMSS` with your actual backup directory name.

### Step 2: Confirm Restore
The script will prompt for confirmation before proceeding.

### Step 3: Monitor Progress
The restore process will:
1. Stop any existing demo service
2. Restore application files
3. Restore AI models
4. Restore systemd service
5. Load kernel modules
6. Enable and start the service

### Step 4: Verify
After restore completes, check:
```bash
adb shell systemctl status imdt-deepx-demo
adb shell journalctl -u imdt-deepx-demo -f
```

The demo should be running and displaying on the connected screen.

---

## Manual Backup (Alternative Method)

If you prefer manual backup or need to create a full system image:

### Option 1: ADB Pull Method
```powershell
# Backup critical directories
adb pull /usr/share/dx-stream ./backup/usr_share_dx-stream
adb pull /opt ./backup/opt
adb pull /lib/systemd/system/imdt-deepx-demo.service ./backup/
adb pull /lib/modules/5.15.148-qki-consolidate/extra ./backup/kernel_modules
```

### Option 2: Create Tarball on Board
```bash
# On the board
adb shell
cd /tmp
tar czf deepx_backup.tar.gz \
    /usr/share/dx-stream \
    /opt \
    /lib/systemd/system/imdt-deepx-demo.service \
    /lib/modules/$(uname -r)/extra

# Pull to PC
exit
adb pull /tmp/deepx_backup.tar.gz ./
```

### Option 3: Full System Image (Advanced)
```bash
# Create full partition backup (requires root)
adb shell dd if=/dev/block/bootdevice/by-name/system of=/tmp/system.img
adb pull /tmp/system.img ./
```

---

## Troubleshooting

### Service Won't Start
```bash
# Check service status
adb shell systemctl status imdt-deepx-demo

# View detailed logs
adb shell journalctl -u imdt-deepx-demo -n 100

# Check if kernel module loaded
adb shell lsmod | grep dxrt_driver

# Manually load module
adb shell modprobe dxrt_driver
```

### Missing Dependencies
```bash
# Check GStreamer plugins
adb shell gst-inspect-1.0 | grep -E 'deepx|qti'

# Verify models exist
adb shell ls -lh /opt/*.tflite

# Check device nodes
adb shell ls -la /dev/fastrpc* /dev/adsprpc*
```

### Display Issues
```bash
# Check Weston compositor
adb shell ps aux | grep weston

# Verify Wayland display
adb shell echo $WAYLAND_DISPLAY

# Check graphics service
adb shell systemctl status init_qti_graphics.service
```

### Permission Issues
```bash
# Remount filesystem as read-write
adb shell mount -o rw,remount /

# Fix permissions
adb shell chmod +x /usr/share/dx-stream/imdt-deepx-demo.sh
adb shell chmod 644 /lib/systemd/system/imdt-deepx-demo.service
```

---

## Service Management Commands

### Start/Stop/Restart
```bash
adb shell systemctl start imdt-deepx-demo
adb shell systemctl stop imdt-deepx-demo
adb shell systemctl restart imdt-deepx-demo
```

### Enable/Disable Auto-start
```bash
adb shell systemctl enable imdt-deepx-demo   # Auto-start on boot
adb shell systemctl disable imdt-deepx-demo  # Disable auto-start
```

### View Logs
```bash
# Live logs
adb shell journalctl -u imdt-deepx-demo -f

# Last 50 lines
adb shell journalctl -u imdt-deepx-demo -n 50

# Since last boot
adb shell journalctl -u imdt-deepx-demo -b
```

---

## Important Notes

### Before Making Changes
1. **Always create a backup first** using `backup_board.ps1`
2. **Test the restore process** on a clean board if possible
3. **Document any custom modifications** you make

### System Requirements
- Qualcomm QCS8550 board with BSP v2.0.0 or compatible
- Kernel version: 5.15.148-qki-consolidate (or compatible)
- Weston compositor running
- QMMF server service running
- FastRPC device nodes available

### What's NOT Backed Up
- Base system files (kernel, bootloader, BSP)
- System libraries (assumed to be part of BSP)
- User data in `/data` (unless explicitly added)
- Network configurations
- User accounts and passwords

### Storage Considerations
- Backup requires ~200 MB on PC
- Restore requires ~200 MB free space on board
- Ensure `/data` partition has sufficient space

---

## Next Steps

After backing up the current demo:

1. **Verify backup completeness** - Check all files are present
2. **Test restore on clean board** (if you have a spare)
3. **Develop your new project** - You can now safely modify the board
4. **Keep backup safe** - Store in multiple locations
5. **Document changes** - Track what you modify from the original

---

## Support

For issues or questions:
- Check logs: `adb shell journalctl -u imdt-deepx-demo -f`
- Review manifest: `cat board_backup_*/BACKUP_MANIFEST.txt`
- Verify ADB connection: `adb devices`
- Check board status: `adb shell systemctl status`

---

**Created**: 2026-05-25  
**Board**: Qualcomm QCS8550 (Device ID: a9ef4ffe)  
**Application**: IMDT DeepX Demo v1.0
