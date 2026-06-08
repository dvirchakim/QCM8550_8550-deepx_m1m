# Qualcomm QCS8550 + DeepX M1 Project

## 🔄 Board Backup & Restore (IMPORTANT!)

**Current Board Status**: Running IMDT DeepX Demo (dual-camera pose estimation + YOLOv8 detection)

### Backup Current Demo (Before Making Changes!)
```powershell
.\backup_board.ps1
```
This creates a complete backup of the running demo application, including:
- DeepX application files
- AI models (YOLOv8, YOLOv5, HRNet)
- Systemd services
- Kernel modules
- GStreamer plugins

**Backup Location**: `board_backup_YYYYMMDD_HHMMSS/`

### Restore Original Demo
```powershell
.\restore_board.ps1 -BackupDir ".\board_backup_YYYYMMDD_HHMMSS"
```

📖 **See `BACKUP_RESTORE_GUIDE.md` for complete documentation**  
📋 **See `BACKUP_SUMMARY.md` for backup details**

---

## 🚀 Quick Setup Guide (New Project)

1.  **Deploy Code**:
    Run the deployment script (ensure you have ADB connected):
    ```powershell
    .\deploy.ps1
    ```

2.  **Environment Setup (On Board)**:
    ```bash
    cd /data/qualcomm_deepx_m1
    
    # Create venv if not exists
    python3 -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    # The board uses 'opkg'. Try installing pip or libraries directly:
    
    # Option A: Install via opkg (Recommended)
    opkg update
    opkg install python3-numpy python3-opencv python3-requests python3-psutil
    
    # Check for Qt support (PyQt6 might formally be python3-pyqt6 or similar)
    opkg list | grep pyqt
    
    # Option B: Install PIP first
    opkg install python3-pip
    # Then: pip install -r requirements.txt
    
    # Option C: If pip/opkg fail, we might need 'get-pip.py'
    Your logs showed `/dev/fastrpc*` is missing. This might block the Qualcomm GenAI demo.
    Try running this on the board to check for errors:
    ```bash
    dmesg | grep fastrpc
    ls -l /dev/adsprpc*
    ```

3.  **Run Agent**:
    ```bash
    export DISPLAY=:0  # If running directly on screen
    python3 main.py
    ```

---

## 📁 Project Files

- `backup_board.ps1` - Backup current board application
- `restore_board.ps1` - Restore backed up application
- `BACKUP_RESTORE_GUIDE.md` - Complete backup/restore documentation
- `BACKUP_SUMMARY.md` - Details of what was backed up
- `deploy.ps1` - Deploy new project to board
- `src/` - Python source code for new project
- `board_backup_*/` - Backup directories (created by backup script)
