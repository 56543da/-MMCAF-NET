# 模型训练数据准备指南

本文档详细说明了如何从原始 DICOM 数据和 XML 标注生成 `MMCAF-Net` 训练所需的全部文件。

**主要更新：**

1. 所有脚本已配置为自动识别路径。只要保持服务器上的目录结构与本地一致，无需手动修改代码中的路径即可直接运行。
2. 以下命令已更新为使用 `mmcafnet` 虚拟环境的 Python 解释器。

## 1. 目录结构说明

请确保服务器上的目录结构类似于以下形式：

```
/path/to/project_root/
├── data/
│   ├── Lung-PET-CT-Dx/                 # 原始DICOM数据
│   ├── Lung-PET-CT-Dx-Annotations.../  # 原始XML标注
│   ├── metadata.csv                    # 临床信息元数据 (必须包含)
│   └── ...                             # 生成的文件将保存在这里
├── MMCAF-Net-main/                     # 代码主目录
│   ├── hdf5_ours.py
│   ├── preprocess_data.py              # 预处理脚本 (已更新，支持自动合并临床数据)
│   ├── merge_data.py                   # 数据合并工具 (用于快速修复/合并临床数据)
│   ├── pkl_read.py
│   └── ...
└── DATA_PREPARATION.md                 # 本文档
```

## 2. 核心文件说明

### 2.1 `data1.hdf5` (影像数据)

* **内容**：存储了所有病人的 CT 扫描切片数据。

* **格式**：HDF5 格式。

* **来源**：由 `MMCAF-Net-main/hdf5_ours.py` 生成。

### 2.2 `G_first_last_nor.csv` (完整元数据)

* **内容**：包含病人的关键标注信息、数据集划分以及**7维临床特征**。

  * **临床特征**：Sex, Age, Weight, T-Stage, N-Stage, M-Stage, Smoking

  * **维度说明**：这些特征对应模型 `KANLayer` 的输入维度 (7)。

* **来源**：

  1. 由 `MMCAF-Net-main/preprocess_data.py` 读取 DICOM、XML 和 `metadata.csv` 完整生成。
  2. 或者由 `MMCAF-Net-main/merge_data.py` 将 `metadata.csv` 合并到现有的 CSV 中。

### 2.3 `series_list_last_AG.pkl` (索引文件)

* **内容**：Python Pickle 序列化对象，包含 `CTPE` 类的实例列表。

* **来源**：由 `MMCAF-Net-main/pkl_read.py` 读取 CSV 生成。

## 3. 运行顺序与命令

在 Windows 下，请使用以下命令（已指定 `mmcafnet` 环境）：

### 第一步：生成影像数据 (HDF5)

使用多线程加速生成 HDF5 文件。

```powershell
# 进入代码目录
cd MMCAF-Net-main

# 默认运行 (自动查找 ../data 下的数据)
E:\conda_envs\mmcafnet\python.exe hdf5_ours.py --threads 8
```

### 第二步：生成完整元数据 (CSV)

解析标注并生成 CSV 文件。此步骤现在会自动读取 `metadata.csv` 并合并临床特征。

**前置条件**：确保 `data/metadata.csv` 存在。

```powershell
# 确保在代码目录下
# cd MMCAF-Net-main

# 默认运行 (自动查找 ../data 下的数据)
# 注意：务必加上 --full 参数以处理所有数据
E:\conda_envs\mmcafnet\python.exe preprocess_data.py --full
```

**\[可选] 快速合并临床数据**
如果您已经生成了 CSV 但缺少临床特征（或者修改了 `metadata.csv`），可以使用 `merge_data.py` 快速合并，而无需重新运行完整的预处理。

```powershell
# 快速合并临床数据到 G_first_last_nor.csv
E:\conda_envs\mmcafnet\python.exe merge_data.py
```

### 第三步：生成索引文件 (PKL)

将 CSV 转换为 Pickle 格式。

```powershell
/path/to/project_root/
├── data/
│   ├── Lung-PET-CT-Dx/                 # 原始DICOM数据
│   ├── Lung-PET-CT-Dx-Annotations.../  # 原始XML标注
│   ├── metadata.csv                    # 临床信息元数据 (必须包含)
│   └── ...                             # 生成的文件将保存在这里
├── MMCAF-Net-main/                     # 代码主目录
│   ├── hdf5_ours.py
│   ├── preprocess_data.py              # 预处理脚本 (已更新，支持自动合并临床数据)
│   ├── merge_data.py                   # 数据合并工具 (用于快速修复/合并临床数据)
│   ├── pkl_read.py
│   └── ...
└── DATA_PREPARATION.md                 # 本文档
```

### 第四步：开始训练

在 Windows 下，直接运行我为您创建的 PowerShell 脚本：

```powershell
# 确保在代码目录下
cd MMCAF-Net-main

# 运行训练脚本
.\train1.ps1
```

该脚本已预配置好：

* **环境自动检测**：优先寻找 `C:\conda_envs\mmcafnet`，若不存在则寻找 `E:\conda_envs\mmcafnet`。

* 数据目录指向 `../data`。

* 训练结果保存在 `../train_result`。

***

*注意：如果在 Linux 服务器上运行，请继续使用原来的* *`train1.sh`，但需要手动修改其中的路径。*
