import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util
import warnings
from sklearn.impute import KNNImputer

# 屏蔽无关警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.helpers is deprecated")

from tqdm import tqdm
from .output_aggregator import OutputAggregator
from cams.grad_cam import GradCAM

###
import shap
import cv2
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import io
import gc

class ModelEvaluator1(object):
    def __init__(self, 
                 dataset_name, 
                 data_loaders, 
                 agg_method = None, 
                 epochs_per_eval = 1):


        self.aggregator=OutputAggregator(agg_method, num_bins=10, num_epochs=5)
        
        self.data_loaders=data_loaders
        self.dataset_name=dataset_name
        self.epochs_per_eval=epochs_per_eval #w
        self.cls_loss_fn= util.optim_util.get_loss_fn(is_classification=True, dataset=dataset_name)
        self.max_eval=None 



    def evaluate(self, model, device, epoch=None, num_epochs=None, table=None):

        #w
        # tab = pd.read_csv('e:/rerun2/data/G_first_last_nor.csv')
        import os
        if table is not None:
            tab = table
        else:
            # 自动从 dataset 获取数据目录，避免硬编码路径错误
            data_dir = self.data_loaders[0].dataset.data_dir
            csv_path = os.path.join(data_dir, 'G_first_last_nor.csv')
            tab = pd.read_csv(csv_path)
            # 已在预处理阶段完成 KNN 插值和归一化，直接使用
        
        metrics, curves={}, {}

        # Grad-CAM 初始化 (只在验证集第一个样本做一次即可，节省时间)
        if not hasattr(self, 'grad_cam'):
             self.grad_cam = GradCAM(model, device, is_binary=True, is_3d=True)
        
        # 目标层：根据 MMCAF_Net 结构，选择图像编码器最后一层特征融合/输出前的层
        # models/MMCAF_Net.py: self.image_encoder = Img_new()
        # models/img_encoder1.py: Img_new 内部包含 self.bfpu2 / self.dcfb1 等层
        # 因此常用目标层应为: image_encoder.bfpu2 / image_encoder.dcfb1
        model_to_scan = model.module if hasattr(model, 'module') else model
        available_layer_names = set(name for name, _ in model_to_scan.named_modules())
        target_layer_candidates = [
            'image_encoder.bfpu2',
            'image_encoder.dcfb1',
            'image_encoder.dcfb2',
            'image_encoder.dcfb3',
        ]
        target_layer = None
        for cand in target_layer_candidates:
            if cand in available_layer_names:
                target_layer = cand
                break
        if target_layer is None:
            print("WARNING: Could not resolve Grad-CAM target layer from candidates; skipping Grad-CAM.")
            target_layer = ''

        #w 还不确定self.data_loaders是不是有多个元素
        sum_loss = []

        model.eval()
        for data_loader in self.data_loaders:
            # Determine if we should perform heavy analysis (SHAP, Grad-CAM, Ablation)
            # 修改为每 20 Epoch 执行一次，或者最后一个 Epoch
            do_analysis = False
            if epoch is not None:
                if epoch % 20 == 0 or (num_epochs is not None and epoch >= num_epochs):
                    do_analysis = True
            
            phase_metrics, phase_curves, sum_every_loss = self._eval_phase(
                model, data_loader, data_loader.phase, device, tab, epoch, do_analysis, target_layer
            )
            metrics.update(phase_metrics)
            curves.update(phase_curves)
            #w
            sum_loss.append(sum_every_loss)
            
        model.train()
        #w
        eval_loss = sum(sum_loss) / len(sum_loss)
        # raise ValueError("eval_loss是{}".format(eval_loss))
        print('eval_loss:', eval_loss)
        ###
        return metrics,curves, eval_loss

    ###
    def _eval_phase(self, model, data_loader, phase, device, table, epoch, do_analysis=False, target_layer='image_encoder.bfpu2'):
        #w
        out = None
        phase_curves = {} # 初始化用于存储分析结果（如 Grad-CAM）的字典


        # 单模态屏蔽评估 (Ablation Study) - 只在 Val/Test 且满足 do_analysis 条件时进行
        ablation_metrics = {}
        if phase in ['val', 'test'] and do_analysis:
            print(f"Running Ablation Study for {phase} (Epoch {epoch})...")
            # 1. Image Only (屏蔽表格: tab全0)
            img_only_loss = self._run_ablation(model, data_loader, device, table, mode='img_only')
            # 2. Table Only (屏蔽图像: img全0)
            tab_only_loss = self._run_ablation(model, data_loader, device, table, mode='tab_only')
            
            # 记录多模态完整 Loss (作为 Baseline)
            baseline_loss = self._run_ablation(model, data_loader, device, table, mode='baseline')
            ablation_metrics[f'{phase}_loss/Multimodal_Baseline'] = baseline_loss
            
            # 计算 Performance Drop (Loss 增加量)
            # Drop > 0 说明屏蔽该模态导致性能下降，证明该模态有用
            ablation_metrics[f'{phase}_loss_drop/Img_Contribution'] = img_only_loss - baseline_loss
            ablation_metrics[f'{phase}_loss_drop/Tab_Contribution'] = tab_only_loss - baseline_loss
            
            ablation_metrics[f'{phase}_loss/ImgOnly'] = img_only_loss
            ablation_metrics[f'{phase}_loss/TabOnly'] = tab_only_loss


        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
        """
        batch_size=data_loader.batch_size

        # Keep track of task-specific records needed for computing overall metrics
    
        records={'keys': [], 'probs': []}
        


     
        num_examples=len(data_loader.dataset)
      

        # Sample from the data loader and record model outputs
        num_evaluated=0

        with tqdm(total=num_examples, unit=' ' + phase) as progress_bar:
            #w
            sum_every_loss = 0

            for img, targets_dict in data_loader:
                if num_evaluated >=num_examples:
                    break

                ###
                ids = [item for item in targets_dict['study_num']]

                tab=[]
                feature_cols = ['Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking']
                for i in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[i]]
                    if patient_row.empty:
                        data = np.zeros(7, dtype=np.float32)
                    else:
                        data = patient_row[feature_cols].iloc[0].values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab).squeeze(1)

                with torch.no_grad():


                    #w process data
                    img = img.to(device)
                    tab = tab.to(device)
                    # 修正：BCEWithLogitsLoss 要求 label 为 Float 类型
                    label = targets_dict['is_abnormal'].to(device).float()

                    
                    #w forward
                    #f,out = model.forward(img)
                    out = model.forward(img,tab)
                    label = label.unsqueeze(1)
                    cls_loss = self.cls_loss_fn(out, label).mean()
                    loss = cls_loss
                    #w
                    sum_every_loss += loss.item()
                    cls_logits = out if out is not None else torch.randn([4, 1])


                    



                #w
                self._record_batch(cls_logits,targets_dict['series_idx'],loss,**records)

                # 可解释性分析 (仅在验证集每个epoch的第一个batch做一次)
                # 优化：只在满足 do_analysis 条件的 Epoch 进行耗时的可视化分析 (Grad-CAM, AttnMap, SHAP)
                if phase == 'val' and num_evaluated == 0 and do_analysis:
                    print(f"DEBUG: Starting heavy analysis (Grad-CAM, SHAP, etc.) for {phase} at Epoch {epoch}...")
                    # 1. Grad-CAM 可视化
                    try:
                        # 目标层：指向图像编码器最后的特征提取层
                        target_layer = target_layer 
                        print(f"DEBUG: Running Grad-CAM on layer: {target_layer}")
                        with torch.enable_grad():
                             if hasattr(self, 'grad_cam') and target_layer:
                                 self.grad_cam.model = model 
                                 self.grad_cam._register_hooks(target_layer)
                                 idx = 0 
                                 # 显式 clone 输入，防止干扰
                                 cam_img = img[idx:idx+1].detach().clone().requires_grad_(True)
                                 cam_tab = tab[idx:idx+1].detach().clone()
                                 
                                 probs, _ = self.grad_cam.forward(cam_img, cam_tab)
                                 self.grad_cam.backward(idx=0)
                                 gcam = self.grad_cam.get_cam(target_layer)
                                 phase_curves[f'{phase}_GradCAM'] = gcam
                                 
                                 # 立即释放钩子并清理梯度
                                 self.grad_cam._release_hooks()
                                 model.zero_grad()
                                 print("DEBUG: Grad-CAM completed and hooks released.")
                    except Exception as e:
                        print(f"ERROR: Grad-CAM failed: {e}")
                        if hasattr(self, 'grad_cam'): self.grad_cam._release_hooks()
                    finally:
                        # 确保无论成功失败都清理缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 2. Attention Matrix 可视化
                    try:
                        if hasattr(model, 'multiscale_fusion'):
                            ca_module = model.multiscale_fusion.cross_attention1
                            if hasattr(ca_module, 'last_attn_map'):
                                print("DEBUG: Logging Attention Map...")
                                attn_map = ca_module.last_attn_map[0, 0] 
                                phase_curves[f'{phase}_AttnMap'] = attn_map.detach().cpu().numpy()
                    except Exception as e:
                        print(f"ERROR: AttnMap log failed: {e}")

                    # 3. SHAP Summary Plot 可视化（使用 KernelExplainer，避免 DeepExplainer 的 backward hook 与 inplace 冲突）
                    try:
                        print("DEBUG: Running SHAP (KernelExplainer) for tabular features...")

                        feature_names = ['Sex', 'Age', 'Weight', 'T', 'N', 'M', 'Smoke']
                        # 控制 KernelExplainer 的计算量，避免一次性生成大量合成样本导致 OOM
                        bg_n = min(8, tab.shape[0])
                        ex_n = min(2, tab.shape[0])

                        x_background = tab[:bg_n].detach().cpu().numpy()
                        x_explain = tab[:ex_n].detach().cpu().numpy()

                        img_fixed = img[:1].detach().clone()

                        def _predict_tab(x_np):
                            x_np = np.asarray(x_np, dtype=np.float32)
                            if x_np.ndim == 1:
                                x_np = x_np[None, :]
                            x_t = torch.tensor(x_np, dtype=torch.float32, device=device)

                            img_fixed_dev = img_fixed.to(device)

                            # 分批推理，防止 KernelExplainer 一次传入过多合成样本导致显存爆炸
                            chunk = 2
                            all_probs = []
                            with torch.inference_mode():
                                for start in range(0, x_t.shape[0], chunk):
                                    end = min(start + chunk, x_t.shape[0])
                                    x_chunk = x_t[start:end]
                                    img_rep = img_fixed_dev.repeat(x_chunk.shape[0], 1, 1, 1, 1)
                                    if torch.cuda.is_available():
                                        with torch.cuda.amp.autocast(dtype=torch.float16):
                                            logits = model.forward(img_rep, x_chunk)
                                    else:
                                        logits = model.forward(img_rep, x_chunk)
                                    probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
                                    all_probs.append(probs)
                            return np.concatenate(all_probs, axis=0)

                        explainer = shap.KernelExplainer(_predict_tab, x_background)
                        shap_values = explainer.shap_values(x_explain, nsamples=50)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]

                        # 显式创建 Figure 并设置大小，避免只有支架
                        fig = plt.figure(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values,
                            x_explain,
                            feature_names=feature_names,
                            show=False,
                            plot_type="bar"
                        )
                        # 强制重绘，确保内容渲染
                        plt.tight_layout()
                        
                        buf = io.BytesIO()
                        # 使用 bbox_inches='tight' 防止内容被裁切
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        shap_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                        shap_img = cv2.imdecode(shap_img, cv2.IMREAD_COLOR)
                        if shap_img is not None:
                            shap_img = cv2.cvtColor(shap_img, cv2.COLOR_BGR2RGB)
                            phase_curves[f'{phase}_SHAP_Summary'] = shap_img
                            print("DEBUG: SHAP summary plot completed.")
                        plt.close(fig) # 显式关闭指定 figure
                        buf.close()
                        del explainer
                    except Exception as e:
                        print(f"ERROR: SHAP plot failed: {e}")
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                progress_bar.update(min(batch_size, num_examples - num_evaluated))
                num_evaluated +=batch_size



        #Map to summary dictionaries
        metrics, curves = self._get_summary_dicts(data_loader, phase, device, **records)
        metrics.update(ablation_metrics) # 合并消融实验结果
        curves.update(phase_curves)      # 合并分析结果图

        ###
        return metrics, curves, sum_every_loss

    def _run_ablation(self, model, data_loader, device, table, mode):
        """快速运行单模态消融评估"""
        total_loss = 0
        count = 0
        limit = 50 # 为了速度，只评估前50个batch
        
        with torch.no_grad():
            for i, (img, targets_dict) in enumerate(data_loader):
                if i >= limit: break
                
                # ... (数据加载逻辑复用) ...
                ids = [item for item in targets_dict['study_num']]
                tab=[]
                feature_cols = ['Sex', 'Age', 'Weight', 'T-Stage', 'N-Stage', 'M-Stage', 'Smoking']
                for k in range(len(targets_dict['study_num'])):
                    patient_row = table[table['NewPatientID'] == ids[k]]
                    if patient_row.empty: data = np.zeros(7, dtype=np.float32)
                    else: data = patient_row[feature_cols].iloc[0].values.astype(np.float32)
                    tab.append(torch.tensor(data, dtype=torch.float32))
                tab=torch.stack(tab).squeeze(1)

                img = img.to(device)
                tab = tab.to(device)
                # 修正：BCEWithLogitsLoss 要求 label 为 Float 类型
                label = targets_dict['is_abnormal'].to(device).float()

                # 实施屏蔽
                if mode == 'img_only':
                    tab = torch.zeros_like(tab)
                elif mode == 'tab_only':
                    img = torch.zeros_like(img)
                # mode == 'baseline' 时不做任何屏蔽，计算完整多模态 Loss

                # 消融实验时不应该触发 Grad-CAM 的 hook 逻辑
                # 虽然我们给 hook 做了保护，但最好还是彻底不记录
                out = model.forward(img, tab)
                label = label.unsqueeze(1)
                loss = self.cls_loss_fn(out, label).mean()
                total_loss += loss.item()
                count += 1
        
        model.zero_grad()
        return total_loss / count if count > 0 else 0

    @staticmethod
    def _record_batch(logits, targets, loss, probs=None, keys=None, loss_meter=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            targets: Batch of ground-truth targets corresponding to the logits.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if probs is not None:
            assert keys is not None, 'Must keep probs and keys lists in parallel'
            with torch.no_grad():
                batch_probs=F.sigmoid(logits)
            probs.append(batch_probs.detach().cpu())

            #Note: `targets` is assumed to hold the keys for these examples
            keys.append(targets.detach().cpu())
        

    def _get_summary_dicts(self, data_loader, phase, device, probs=None, keys=None, loss_meter=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model. E.g. ROC.
        """
        metrics, curves={}, {}

        if probs is not None:
            # If records kept track of individual probs and keys, implied that we need to aggregate them
            assert keys is not None, 'Must keep probs and keys lists in parallel.'
            assert self.aggregator is not None, 'Must specify an aggregator to aggregate probs and keys.'

            # Convert to flat numpy array
            probs=np.concatenate(probs).ravel().tolist()
            keys=np.concatenate(keys).ravel().tolist()

            # Aggregate predictions across each series
            idx2prob=self.aggregator.aggregate(keys, probs, data_loader, phase, device)
            probs, labels=[], []
            for idx, prob in idx2prob.items():
                probs.append(prob)
                labels.append(data_loader.get_series_label(idx))
            probs, labels=np.array(probs), np.array(labels)

            # Update summary dicts
            metrics.update({
                phase + '_' + 'loss': sk_metrics.log_loss(labels, probs, labels=[0, 1])
            })

            # Update summary dicts
            try:
                # Binarize predictions for Acc, Sens, Spec
                preds = (probs >= 0.5).astype(int)
                
                # Calculate metrics
                accuracy = sk_metrics.accuracy_score(labels, preds)
                conf_matrix = sk_metrics.confusion_matrix(labels, preds, labels=[0, 1])
                tn, fp, fn, tp = conf_matrix.ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                f1 = sk_metrics.f1_score(labels, preds)

                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                    phase + '_' + 'Accuracy': accuracy,
                    phase + '_' + 'Sensitivity': sensitivity,
                    phase + '_' + 'Specificity': specificity,
                    phase + '_' + 'PPV': precision,
                    phase + '_' + 'NPV': npv,
                    phase + '_' + 'F1': f1
                })
                curves.update({
                    phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, probs),
                    phase + '_' + 'ROC': sk_metrics.roc_curve(labels, probs),
                    phase + '_' + 'Confusion Matrix': sk_metrics.confusion_matrix(labels, preds)
                })
            except ValueError:
                pass



        return metrics, curves
