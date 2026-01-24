# Windows PowerShell Multi-Experiment Automation Script

# 互斥锁：防止多个后台脚本同时运行
$LockFile = Join-Path $PSScriptRoot "experiment.lock"
if (Test-Path $LockFile) {
    $OldProcessId = Get-Content $LockFile
    if (Get-Process -Id $OldProcessId -ErrorAction SilentlyContinue) {
        Write-Host "Warning: An experiment is already running (PID: $OldProcessId)." -ForegroundColor Red
        Write-Host "To stop it, run: Stop-Process -Id $OldProcessId -Force" -ForegroundColor Yellow
        Write-Host "Then delete the lock file: Remove-Item '$LockFile'" -ForegroundColor Yellow
        exit
    } else {
        Remove-Item $LockFile
    }
}

if ($null -eq $env:RUNNING_IN_BACKGROUND) {
    $env:RUNNING_IN_BACKGROUND = "True"
    $logFile = Join-Path $PSScriptRoot "experiments_progress.log"
    # 清空旧日志
    if (Test-Path $logFile) { Remove-Item $logFile -ErrorAction SilentlyContinue }
    
    # 启动后台进程，直接重定向输出到文件，这比 Start-Transcript 反应快得多
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `"$PSCommandPath *>> '$logFile'`"" -WindowStyle Hidden
    
    Write-Host "Experiments started in background (Unbuffered)!" -ForegroundColor Green
    Write-Host "Streaming logs (Press Ctrl+C to stop viewing, background tasks will continue)..." -ForegroundColor Yellow
    
    while (-not (Test-Path $logFile)) { Start-Sleep -Milliseconds 500 }
    Get-Content $logFile -Wait
    exit
}

# 确保在脚本所在目录下运行
Set-Location $PSScriptRoot
# 强制 Python 不使用输出缓存
$env:PYTHONUNBUFFERED = "1"
# 优化显存碎片处理 (针对 OOM 报错的建议)
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128,expandable_segments:True"

# 记录当前进程 ID 到锁文件
$PID | Out-File $LockFile

# Detect Python Environment (keep consistent with train1.ps1)
$C_PATH = "C:\conda_envs\mmcafnet\python.exe"
$C_MINICONDA_PATH = "C:\ProgramData\miniconda3\envs\mmcafnet\python.exe"
$E_PATH = "E:\conda_envs\mmcafnet\python.exe"

if (Test-Path $C_PATH) {
    $PYTHON_EXE = $C_PATH
} elseif (Test-Path $C_MINICONDA_PATH) {
    $PYTHON_EXE = $C_MINICONDA_PATH
} elseif (Test-Path $E_PATH) {
    $PYTHON_EXE = $E_PATH
} else {
    Write-Error "Could not find mmcafnet environment!"
    Stop-Transcript
    exit 1
}

# Define Experiment List: modify name, lr, opt, batch, epochs
$experiments = @(
    @{ name = "Exp1_SGD_LR1e-4_B32"; lr = "1e-4"; opt = "adam"; batch = "32"; epochs = "50" }
   # @{ name = "Exp2_SGD_LR5e-5_B16"; lr = "1e-4"; opt = "sgd"; batch = "32"; epochs = "100" }
   # @{ name = "Exp3_SGD_LR1e-3_B32";  lr = "1e-4"; opt = "sgd";  batch = "4"; epochs = "200" }
)

foreach ($exp in $experiments) {
    Write-Host "`n" + ("="*60) -ForegroundColor Cyan
    Write-Host "Starting Experiment: $($exp.name)" -ForegroundColor Cyan
    Write-Host "Config: Optimizer=$($exp.opt), LR=$($exp.lr), BatchSize=$($exp.batch), Epochs=$($exp.epochs)" -ForegroundColor Cyan
    Write-Host ("="*60) -ForegroundColor Cyan

    $trainArgs = @(
        "--data_dir=../data"
        "--save_dir=../train_result"
        "--name=$($exp.name)"
        "--model=MMCAF_Net"
        "--batch_size=$($exp.batch)"
        "--gpu_ids=0"
        "--iters_per_print=$($exp.batch)"
        "--iters_per_visual=8000"
        "--learning_rate=$($exp.lr)"
        "--lr_decay_step=600000"
        "--lr_scheduler=cosine_warmup"
        "--num_epochs=$($exp.epochs)"
        "--num_slices=12"
        "--weight_decay=1e-2"
        "--phase=train"
        "--agg_method=max"
        "--best_ckpt_metric=val_loss"
        "--crop_shape=192,192"
        "--cudnn_benchmark=False"
        "--dataset=pe"
        "--do_classify=True"
        "--epochs_per_eval=1"
        "--epochs_per_save=1"
        "--fine_tune=False"
        "--fine_tuning_boundary=classifier"
        "--fine_tuning_lr=1e-2"
        "--include_normals=True"
        "--lr_warmup_steps=10000"
        "--model_depth=50"
        "--num_classes=1"
        "--num_visuals=8"
        "--num_workers=4"
        "--optimizer=$($exp.opt)"
        "--pe_types=['central','segmental']"
        "--resize_shape=192,192"
        "--sgd_dampening=0.9"
        "--sgd_momentum=0.9"
        "--use_pretrained=False"
    )

    # --- 断点续训 (Auto Resume) 逻辑 ---
    # 设置为 $false 以强制从头开始训练
    $enableResume = $true
    
    # 增加手动指定后缀的功能（如 155855）
    $manualSuffix = "" # 如果要指定特定目录，请在此填入后缀，例如 "155855"

    # 查找该实验名称对应的目录
    if ($manualSuffix -ne "") {
        $searchPattern = "../train_result/$($exp.name)_*$manualSuffix"
    } else {
        $searchPattern = "../train_result/$($exp.name)_20*"
    }

    if ($enableResume -and (Test-Path $searchPattern)) {
        $latestDir = Get-ChildItem -Path $searchPattern -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        
        if ($latestDir) {
            Write-Host "Checking for checkpoints in $($latestDir.FullName)..." -ForegroundColor Gray
            # 查找最新的 epoch_X.pth.tar (排除 best.pth.tar)
            $latestCkpt = Get-ChildItem -Path "$($latestDir.FullName)/epoch_*.pth.tar" | 
                Sort-Object { [int]($_.Name -replace 'epoch_','' -replace '.pth.tar','') } -Descending | 
                Select-Object -First 1
                
            if ($latestCkpt) {
                Write-Host "Found existing checkpoint: $($latestCkpt.Name). Resuming training..." -ForegroundColor Yellow
                $trainArgs += "--ckpt_path=$($latestCkpt.FullName)"
            }
        }
    }
    # ------------------------------------

    # 执行训练
    & $PYTHON_EXE train1.py @trainArgs

    Write-Host "Experiment $($exp.name) finished.`n" -ForegroundColor Green
    
    # 每个实验结束后等待 10 秒，让显存彻底释放
    Write-Host "Waiting 10s for GPU memory cleanup..." -ForegroundColor Gray
    Start-Sleep -Seconds 10
}

Write-Host "All experiments completed!" -ForegroundColor Magenta
# 完成后删除锁文件
if (Test-Path $LockFile) { Remove-Item $LockFile }
