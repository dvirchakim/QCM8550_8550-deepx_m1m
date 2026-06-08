# Backup Summary - Qualcomm QCS8550 Board

## ✅ Backup Completed Successfully

**Date**: 2026-05-25 17:34:08  
**Device**: a9ef4ffe  
**Backup Location**: `board_backup_20260525_173350`

---

## 📦 What Was Backed Up

### 1. **DeepX Demo Application** (102 files)
- **Location**: `/usr/share/dx-stream/`
- **Main Script**: `imdt-deepx-demo.sh` (4.2 KB)
- **Configs**: 77 configuration files for pose estimation and detection
- **Libraries**: 21 DeepX library files
- **Assets**: Logo and resources

### 2. **AI Models** (28 files, ~126 MB)
- **Location**: `/opt/`
- **Models**:
  - YOLOv8 Detection (Quantized) - 3.4 MB
  - YOLOv5 Pose - 21.8 MB
  - HRNet Pose (Quantized) - 29.4 MB
  - Inception v3 - 24.4 MB
  - MiDAS Depth - 17.7 MB
  - Face Detection/Recognition models
  - DeepLabV3+ Segmentation
  - And more...
- **Label Files**: COCO, YOLO, pose estimation labels

### 3. **Systemd Services** (2 files)
- `imdt-deepx-demo.service` - Main demo auto-start service
- Service dependencies documented

### 4. **Kernel Modules** (33 files)
- **DeepX Runtime Driver**: `dxrt_driver.ko`
- **Kernel Version**: 5.15.148-qki-consolidate
- **Location**: `/lib/modules/5.15.148-qki-consolidate/extra/`

### 5. **GStreamer Plugins** (228 files)
- **Location**: `/usr/lib/gstreamer-1.0/`
- **DeepX Plugins**:
  - `libgstdxpreprocess.so` - Preprocessing
  - `libgstdxinfer.so` - Inference
  - `libgstdxpostprocess.so` - Post-processing
  - `libgstdxosd.so` - On-screen display
- **Qualcomm Plugins**:
  - `libgstqtiqmmfsrc.so` - Camera source
  - `libgstqtivtransform.so` - Video transform
  - `libgstqtivcomposer.so` - Video compositor
  - And many more...

### 6. **System Information** (8 files)
- Kernel version and system info
- Loaded kernel modules list
- Running processes snapshot
- Active services list
- Disk usage and mount points
- GStreamer dependencies

### 7. **Device Nodes Info** (4 files)
- FastRPC devices (`/dev/fastrpc*`)
- ADSP RPC devices (`/dev/adsprpc*`)
- Video devices (`/dev/video*`)
- DRI devices (`/dev/dri/*`)
- All system symlinks

---

## 🎯 Current Running Demo Details

**Service**: `imdt-deepx-demo.service`  
**Status**: Active (running since boot)  
**Auto-start**: Enabled

### Pipeline Architecture
```
Camera 0 (1920x1080@15fps)
  ↓
  ├─→ YOLOv5 Pose Estimation → OSD → Display Grid (0,0)
  └─→ YOLOv8 Object Detection → Display Grid (0,300)

Camera 1 (1920x1080@15fps)
  ↓
  └─→ YOLOv5 Pose Estimation → OSD → Display Grid (512,0)

Raw Camera Feed → Display Grid (512,300)
Logo Image → Display Grid (512,300)
```

### Display Layout (5 panels)
- **Panel 1** (0, 0): Camera 0 with pose estimation
- **Panel 2** (512, 0): Camera 1 with pose estimation
- **Panel 3** (0, 300): Camera 0 with object detection
- **Panel 4** (512, 300): Raw camera feed
- **Panel 5** (512, 300): Logo overlay

### Hardware Acceleration
- **HTP (Hexagon Tensor Processor)**: Performance mode 2
- **FastRPC**: DSP communication
- **Wayland**: Display compositor
- **QMMF**: Qualcomm Multimedia Framework

---

## 📋 Files Created

1. **`backup_board.ps1`** - Automated backup script
2. **`restore_board.ps1`** - Automated restore script
3. **`BACKUP_RESTORE_GUIDE.md`** - Complete documentation
4. **`board_backup_20260525_173350/`** - Backup directory with all files

---

## 🔄 How to Restore

### Quick Restore
```powershell
.\restore_board.ps1 -BackupDir ".\board_backup_20260525_173350"
```

### What Restore Does
1. Stops existing demo service
2. Restores all application files
3. Restores AI models
4. Restores systemd service configuration
5. Loads kernel modules
6. Enables and starts the service

---

## ⚠️ Important Notes

### What's Included
✅ Complete DeepX demo application  
✅ All AI models and configurations  
✅ Systemd service files  
✅ Kernel modules (dxrt_driver)  
✅ GStreamer plugins  
✅ System state documentation  

### What's NOT Included
❌ Base system files (kernel, bootloader)  
❌ BSP (Board Support Package)  
❌ System libraries (assumed in BSP)  
❌ Network configurations  
❌ User accounts  

### Prerequisites for Restore
- Qualcomm QCS8550 board with BSP v2.0.0+
- Kernel 5.15.148-qki-consolidate (or compatible)
- Weston compositor installed
- QMMF server service available
- ADB connection established

---

## 🎉 You're Ready!

You can now safely:
1. ✅ Develop your new project
2. ✅ Modify the board configuration
3. ✅ Install new applications
4. ✅ Experiment with different setups

**If anything goes wrong**, simply run the restore script to get back to this working state!

---

## 📞 Quick Commands

### Check Demo Status
```bash
adb shell systemctl status imdt-deepx-demo
```

### View Live Logs
```bash
adb shell journalctl -u imdt-deepx-demo -f
```

### Stop Demo
```bash
adb shell systemctl stop imdt-deepx-demo
```

### Start Demo
```bash
adb shell systemctl start imdt-deepx-demo
```

### Disable Auto-start
```bash
adb shell systemctl disable imdt-deepx-demo
```

---

**Backup Size**: ~126 MB (models) + ~5 MB (configs/scripts) = **~131 MB total**  
**Backup Time**: ~2-3 minutes  
**Restore Time**: ~3-5 minutes  

**Status**: ✅ **BACKUP COMPLETE - READY FOR NEW PROJECT DEVELOPMENT**
