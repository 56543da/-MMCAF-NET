import os
import glob
import numpy as np
import pandas as pd
import pydicom
import xml.etree.ElementTree as ET
import random
import argparse
from tqdm import tqdm

# 自动识别路径: 基于当前脚本所在目录
# 脚本位于 e:\rerun2\MMCAF-Net-main\preprocess_data.py
# 数据位于 e:\rerun2\data (即脚本上一级目录的 data)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) # e:\rerun2
DEFAULT_DCM_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Lung-PET-CT-Dx')
DEFAULT_XML_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020', 'Annotation')
DEFAULT_METADATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'metadata.csv')
DEFAULT_OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'G_first_last_nor.csv')

# 设置随机种子以保证可复现性
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess data metadata')
    parser.add_argument('--dcm_root', type=str, default=DEFAULT_DCM_ROOT, help='Root directory of DICOM files')
    parser.add_argument('--xml_root', type=str, default=DEFAULT_XML_ROOT, help='Root directory of XML annotations')
    parser.add_argument('--meta_csv', type=str, default=DEFAULT_METADATA_CSV, help='Path to raw clinical metadata CSV')
    parser.add_argument('--output_csv', type=str, default=DEFAULT_OUTPUT_CSV, help='Output path for the metadata CSV')
    parser.add_argument('--demo', action='store_true', help='Process partial dataset for demo (default is full dataset)')
    return parser.parse_args()

from sklearn.impute import KNNImputer
import concurrent.futures

# ... imports ...

def process_patient_wrapper(args):
    """Wrapper for parallel processing"""
    patient_dir_name, dcm_root, xml_root = args
    patient_path = os.path.join(dcm_root, patient_dir_name)
    try:
        success, meta = process_patient(patient_dir_name, patient_path, xml_root)
        return patient_dir_name, success, meta
    except Exception:
        return patient_dir_name, False, None

def load_and_process_metadata(meta_path):
    """读取并处理临床表格数据 (包含 KNN 插值和归一化)"""
    if not os.path.exists(meta_path):
        print(f"Warning: Metadata file not found at {meta_path}")
        return {}

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return {}
    
    # 映射字典
    sex_map = {'M': 1, 'F': 0}
    t_map = {'is': 0, '1': 1, '1a': 1, '1b': 2, '1c': 3, '2': 4, '2a': 4, '2b': 5, '3': 6, '4': 7}
    m_map = {'0': 0, '1': 1, '1a': 1, '1b': 2, '1c': 3, '2': 4, '3': 5}
    
    # 1. 预构建数值型 DataFrame
    processed_data = []
    pids = []
    
    for _, row in df.iterrows():
        pid = str(row['NewPatientID']).strip()
        pids.append(pid)
        
        # Sex
        sex = sex_map.get(row['Sex'], np.nan) # 使用 NaN 标记缺失
        
        # Age
        try: age = float(row['Age'])
        except: age = np.nan
            
        # Weight
        try: weight = float(row['weight (kg)'])
        except: weight = np.nan
            
        # T-Stage
        t_val = str(row['T-Stage']).strip()
        t_stage = t_map.get(t_val, np.nan)
        
        # N-Stage
        try: n_stage = int(row['N-Stage'])
        except: n_stage = np.nan
            
        # M-Stage
        m_col = 'Ｍ-Stage' if 'Ｍ-Stage' in row else 'M-Stage'
        m_val = str(row.get(m_col, '0')).strip()
        m_stage = m_map.get(m_val, np.nan)
        
        # Smoking
        try: smoking = int(row['Smoking History'])
        except: smoking = np.nan
            
        processed_data.append([sex, age, weight, t_stage, n_stage, m_stage, smoking])
    
    # 转为 DataFrame
    cols = ['Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking']
    data_df = pd.DataFrame(processed_data, columns=cols)
    
    # 2. KNN 插值
    print("Applying KNN Imputation...")
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(data_df)
    data_df = pd.DataFrame(data_imputed, columns=cols)
    
    # 3. 归一化 (Min-Max Normalization)
    # 修正：只对连续变量 (Age, Weight) 进行 Min-Max 归一化
    # T-Stage, N-Stage, M-Stage 是序数变量，为防止 KAN/NN 数值不稳定，
    # 我们将其除以 10.0 进行缩放，既保留了整数等级含义，又限制在 [0,1] 范围内
    print("Applying Normalization...")
    cols_to_norm = ['Age', 'Weight']
    for col in cols_to_norm:
        if col in data_df.columns:
            min_val = data_df[col].min()
            max_val = data_df[col].max()
            if max_val > min_val:
                data_df[col] = (data_df[col] - min_val) / (max_val - min_val)
                
    # 对 TNM 进行固定缩放 (Scale by 10)
    ordinal_cols = ['T-Stage', 'N-Stage', 'M-Stage']
    for col in ordinal_cols:
        if col in data_df.columns:
            data_df[col] = data_df[col] / 10.0
    
    # 重组为字典
    final_meta = {}
    for pid, values in zip(pids, data_df.values):
        final_meta[pid] = values.tolist()
        
    return final_meta

def parse_xml_bbox(xml_path):
    """解析XML获取bbox和SOPInstanceUID"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        uid = os.path.basename(xml_path).replace('.xml', '')
        
        obj = root.find('object')
        if obj is not None:
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                return uid, [xmin, ymin, xmax, ymax]
    except Exception as e:
        pass
    return None, None

def process_patient(patient_id, patient_dir, xml_root):
    """处理单个病人的数据"""
    xml_dir = os.path.join(xml_root, patient_id)
    if not os.path.exists(xml_dir):
        short_id = patient_id.replace('Lung_Dx-', '')
        xml_dir = os.path.join(xml_root, short_id)
        if not os.path.exists(xml_dir):
            return None, None

    xml_map = {}
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for xml_file in xml_files:
        uid, bbox = parse_xml_bbox(xml_file)
        if uid and bbox:
            xml_map[uid] = bbox

    if not xml_map:
        return None, None

    best_series_files = []
    best_series_matches = 0
    best_series_bboxes = []
    
    # 获取所有子目录
    sub_dirs = []
    for root, dirs, files in os.walk(patient_dir):
        if any(f.lower().endswith('.dcm') for f in files):
            sub_dirs.append((root, files))

    # 如果有多个序列，显示一个简易进度
    for root, files in sub_dirs:
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        current_series_files = []
        matches = 0
        current_bboxes = []
        
        for dcm_file in dcm_files:
            dcm_path = os.path.join(root, dcm_file)
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                sop_uid = ds.SOPInstanceUID
                instance_num = int(ds.InstanceNumber)
                
                info = {
                    'path': dcm_path,
                    'instance_num': instance_num,
                    'sop_uid': sop_uid
                }
                current_series_files.append(info)
                
                if sop_uid in xml_map:
                    matches += 1
                    current_bboxes.append(xml_map[sop_uid])
            except:
                pass
        
        if matches > best_series_matches:
            best_series_matches = matches
            best_series_files = current_series_files
            best_series_bboxes = current_bboxes
        elif matches > 0 and matches == best_series_matches:
             if len(current_series_files) > len(best_series_files):
                 best_series_files = current_series_files
                 best_series_bboxes = current_bboxes

    if best_series_matches == 0:
        return None, None

    best_series_files.sort(key=lambda x: x['instance_num'])
    num_slice = len(best_series_files)
    
    abnormal_indices = []
    for idx, info in enumerate(best_series_files):
        if info['sop_uid'] in xml_map:
            abnormal_indices.append(idx)
            
    if not abnormal_indices:
        return None, None
        
    first_appear = min(abnormal_indices)
    last_appear = max(abnormal_indices)
    avg_bbox = np.mean(np.array(best_series_bboxes), axis=0).astype(int).tolist()
    
    return True, {
        'num_slice': num_slice,
        'first_appear': first_appear,
        'last_appear': last_appear,
        'avg_bbox': avg_bbox
    }

def main():
    args = parse_args()
    dcm_root = args.dcm_root
    xml_root = args.xml_root
    output_csv = args.output_csv
    meta_csv = args.meta_csv
    csv_rows = []
    
    if not os.path.exists(dcm_root):
        print(f"Error: DICOM root directory not found: {dcm_root}")
        return

    # Load clinical metadata
    print(f"Loading clinical metadata from {meta_csv}...")
    patient_meta = load_and_process_metadata(meta_csv)
    print(f"Loaded metadata for {len(patient_meta)} patients.")

    patient_dirs = [d for d in os.listdir(dcm_root) if os.path.isdir(os.path.join(dcm_root, d))]
    
    if args.demo:
        patient_dirs = patient_dirs[:20]
        print("Running in DEMO mode (first 20 patients).")
    else:
        print("Running in FULL mode (processing all patients).")
    
    print(f"Found {len(patient_dirs)} patients. Starting preprocessing...")
    
    # Prepare arguments for parallel processing
    tasks = [(d, dcm_root, xml_root) for d in patient_dirs]
    
    # Use ProcessPoolExecutor for parallel processing
    # Max workers set to default (usually number of processors)
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use tqdm to show progress bar
        results = list(tqdm(executor.map(process_patient_wrapper, tasks), total=len(tasks), desc="Overall Progress"))

    # Process results
    for patient_dir_name, success, meta in results:
        short_id = patient_dir_name.replace('Lung_Dx-', '')
        
        if success:
            label = 0 if short_id.startswith('A') else 1
            rand_val = random.random()
            # 采用 7:3 比例进行划分，不使用固定样本量
            # 30% 全部标记为 val，方便训练时直接评估
            if rand_val < 0.7:
                parser = 'train'
            else:
                parser = 'val'
            
            # Get clinical data, default to zeros if missing
            clin_data = patient_meta.get(short_id, [0, 0, 0, 0, 0, 0, 0])
            
            # Construct row
            row_dict = {
                'NewPatientID': short_id,
                'Sex': clin_data[0],
                'Age': clin_data[1],
                'Weight': clin_data[2],
                'T-Stage': clin_data[3],
                'N-Stage': clin_data[4],
                'M-Stage': clin_data[5],
                'Smoking': clin_data[6],
                'label': label,
                'parser': parser,
                'num_slice': meta['num_slice'],
                'first_appear': meta['first_appear'],
                'avg_bbox': str(meta['avg_bbox']),
                'last_appear': meta['last_appear']
            }
            csv_rows.append(row_dict)
            
    df = pd.DataFrame(csv_rows)
    
    # --- Random Over Sampling (ROS) Logic ---
    print("\nApplying Random Over Sampling (ROS) to Training Set...")
    # Separate Train and Val
    train_df = df[df['parser'] == 'train']
    val_df = df[df['parser'] == 'val']
    
    # Analyze Train Distribution
    train_0 = train_df[train_df['label'] == 0]
    train_1 = train_df[train_df['label'] == 1]
    
    print(f"Original Train Distribution: Class 0 (Adeno) = {len(train_0)}, Class 1 (SCC) = {len(train_1)}")
    
    # Target count is the majority class size (or fixed 198 as per paper, but balancing is safer)
    target_count = max(len(train_0), len(train_1))
    
    # Oversample Class 1 (SCC) if needed
    if len(train_1) < target_count:
        print(f"Oversampling Class 1 (SCC) from {len(train_1)} to {target_count}...")
        train_1_over = train_1.sample(n=target_count, replace=True, random_state=42)
        train_df_balanced = pd.concat([train_0, train_1_over])
    else:
        train_df_balanced = pd.concat([train_0, train_1])
        
    # Oversample Class 0 if needed (rare case in this dataset, but good for robustness)
    if len(train_0) < target_count:
         print(f"Oversampling Class 0 (Adeno) from {len(train_0)} to {target_count}...")
         train_0_over = train_0.sample(n=target_count, replace=True, random_state=42)
         train_df_balanced = pd.concat([train_0_over, train_1] if len(train_1) >= target_count else [train_0_over, train_1_over])

    # Shuffle training data
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Train Distribution: Class 0 = {len(train_df_balanced[train_df_balanced['label']==0])}, Class 1 = {len(train_df_balanced[train_df_balanced['label']==1])}")
    
    # Combine with Validation set
    df = pd.concat([train_df_balanced, val_df]).reset_index(drop=True)
    # ----------------------------------------

    # Reorder columns to ensure features are at indices 1-7
    cols = ['NewPatientID', 'Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking', 
            'label', 'parser', 'num_slice', 'first_appear', 'avg_bbox', 'last_appear']
    df = df[cols]
    df.to_csv(output_csv, index=False)
    
    # 打印详细统计信息
    print("\n" + "="*50)
    print("Metadata Generation Complete!")
    print(f"Output saved to: {output_csv}")
    print(f"Total patients processed: {len(df)}")
    print("\nDataset Split Statistics:")
    print(df['parser'].value_counts())
    print("\nLabel Distribution:")
    print(df['label'].value_counts().rename({0: 'Normal', 1: 'Abnormal'}))
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
