import os
import pydicom
import h5py
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

# 自动识别路径: 基于当前脚本所在目录
# 假设脚本位于 e:\rerun2\MMCAF-Net-main\hdf5_ours.py
# 数据位于 e:\rerun2\data
# 需要回退一级目录找到 data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # e:\rerun2
DEFAULT_DCM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Lung-PET-CT-Dx')
DEFAULT_OUTPUT_H5 = os.path.join(PROJECT_ROOT, 'data', 'data1.hdf5')

# 线程锁，用于同步写入HDF5文件
h5_lock = Lock()

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DICOM to HDF5 with multi-threading')
    parser.add_argument('--dcm_root', type=str, default=DEFAULT_DCM_ROOT, help='Root directory of DICOM files')
    parser.add_argument('--output_h5', type=str, default=DEFAULT_OUTPUT_H5, help='Output path for the HDF5 file')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads to use')
    return parser.parse_args()

def process_patient_directory(dirpath, filenames, h5f):
    """处理单个目录下的DICOM文件并写入HDF5"""
    data_list = []
    
    # 筛选并读取DICOM文件
    dcm_files = [f for f in filenames if f.lower().endswith('.dcm') or f.lower().endswith('.dicom')]
    if not dcm_files:
        return

    # 按文件名排序，尝试保证顺序（虽然文件名不一定对应切片顺序，但在本脚本逻辑中暂且如此）
    # 注意：更严谨的做法是读取InstanceNumber进行排序，但为了保持与原代码逻辑相似且保证速度，这里可能需要权衡
    # 原代码没有排序，这里我们加上简单的文件名排序
    dcm_files.sort()

    for filename in dcm_files:
        file_path = os.path.join(dirpath, filename)
        try:
            ds = pydicom.dcmread(file_path)
            pixel_data = ds.pixel_array
            data_list.append(pixel_data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not data_list:
        return

    data_list = np.array(data_list)
    
    # 检查图像尺寸
    if data_list.shape[1] != 512:
        return

    # 提取Patient ID
    path_parts = dirpath.split(os.sep)
    patient_id = None
    for part in path_parts:
        if 'Lung_Dx-' in part:
            patient_id = part.split('-')[-1] # A0001
            break
    
    if not patient_id:
        # print(f"Warning: Could not extract patient ID from {dirpath}")
        return

    # 写入HDF5 (加锁)
    with h5_lock:
        if patient_id in h5f:
            existing_shape = h5f[patient_id].shape
            if data_list.shape[0] > existing_shape[0]:
                print(f"Updating {patient_id}: {existing_shape} -> {data_list.shape}")
                del h5f[patient_id]
                h5f.create_dataset(patient_id, data=data_list, compression='gzip')
            else:
                # print(f"Skipping {patient_id}: New data has fewer or equal slices")
                pass
        else:
            print(f"Adding {patient_id}: {data_list.shape}")
            h5f.create_dataset(patient_id, data=data_list, compression='gzip')

def main():
    args = parse_args()
    
    dcm_root = args.dcm_root
    output_h5 = args.output_h5
    num_threads = args.threads
    
    print(f"Scanning directory: {dcm_root}")
    print(f"Output HDF5: {output_h5}")
    print(f"Threads: {num_threads}")

    # 收集所有需要处理的目录
    tasks = []
    for dirpath, dirnames, filenames in os.walk(dcm_root):
        if 'ALPHA' in dirpath:
            continue
        if any(f.lower().endswith('.dcm') for f in filenames):
            tasks.append((dirpath, filenames))
    
    print(f"Found {len(tasks)} directories with DICOM files.")

    # 打开HDF5文件
    # mode='a' Read/write if exists, create otherwise
    with h5py.File(output_h5, 'a') as h5f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for dirpath, filenames in tasks:
                futures.append(executor.submit(process_patient_directory, dirpath, filenames, h5f))
            
            # 使用tqdm显示进度
            for _ in tqdm(futures, total=len(futures), desc="Processing"):
                _.result() # 等待任务完成并捕获异常

    print("Done!")

if __name__ == '__main__':
    main()
