$REMOTE_DIR = "/data/qualcomm_deepx_m1"
$ADB        = "C:\platform-tools\platform-tools\adb.exe"
$DEVICE     = "a9ef4ffe"

function Adb { param([string]$Cmd) cmd /c "$ADB -s $DEVICE $Cmd" }

Write-Host "Packaging project..." -ForegroundColor Cyan
python package.py

Write-Host "Pushing to device..." -ForegroundColor Cyan
Adb "push deploy.zip /data/local/tmp/deploy.zip"

Write-Host "Cleaning and Extracting on device..." -ForegroundColor Cyan
Adb "shell mkdir -p $REMOTE_DIR"
Adb "shell rm -rf $REMOTE_DIR/src $REMOTE_DIR/assets $REMOTE_DIR/config.py $REMOTE_DIR/main.py $REMOTE_DIR/requirements.txt"
Adb "shell unzip -o /data/local/tmp/deploy.zip -d $REMOTE_DIR"
Adb "shell rm /data/local/tmp/deploy.zip"

Write-Host "Deployment Complete." -ForegroundColor Green
Write-Host "To run:"
Write-Host "  adb -s $DEVICE shell"
Write-Host "  cd $REMOTE_DIR && python3 main.py"
