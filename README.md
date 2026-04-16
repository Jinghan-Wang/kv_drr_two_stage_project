# KV→DRR→Spine (single `pth`, two-stage total network)

这是一个基于 PyTorch 的 2D 工程示例，适用于：

- 输入：`KV`、`DRR`、`spine_only`，均为 `nii.gz`
- 前处理：
  - `KV` 构造 4 通道：`原始`、`2000-Max`、`3000-Max`、`4000-Max`
  - `DRR` 构造 4 通道：`原始`、`1000-Max`、`2000-Max`、`3000-Max`
- 网络：
  - Stage1：`KV(4ch) -> fake_DRR(1ch)`
  - Stage2：`[fake_DRR, fake_DRR_1000, fake_DRR_2000, fake_DRR_3000](4ch) -> pred_spine(1ch)`
- 训练：
  - Stage1 用真实 `DRR` 监督
  - Stage2 用真实 `spine_only` 监督
  - 可配置是否将 Stage2 的梯度回传给 Stage1（`detach_stage1_to_stage2`）
- 验证：保存 `fake_DRR.nii.gz`、`pred_spine.nii.gz`、输入 `KV/DRR/spine` 的 `nii.gz`

## 目录结构

```text
project/
├── README.md
├── requirements.txt
├── train.py
├── validate.py
├── src/
│   ├── configs/
│   │   └── default.yaml
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── total_net.py
│   │   └── unet2d.py
│   └── utils/
│       ├── io.py
│       ├── losses.py
│       ├── thresholds.py
│       └── misc.py
└── (your csv and data)
```

## CSV 格式

训练/验证 CSV 文件至少包含三列：

```csv
kv,drr,spine
/path/to/case001_kv.nii.gz,/path/to/case001_drr.nii.gz,/path/to/case001_spine.nii.gz
/path/to/case002_kv.nii.gz,/path/to/case002_drr.nii.gz,/path/to/case002_spine.nii.gz
...
```

## 运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

```bash
python train.py --config src/configs/default.yaml
```

### 只做验证

```bash
python validate.py \
  --config src/configs/default.yaml \
  --checkpoint /path/to/checkpoints/best.pt
```

## 配置说明

见 `src/configs/default.yaml`。

特别注意：

- `data.intensity_scale`：用于把原始强度缩放到较稳定的范围（例如 4095）
- `data.kv_thresholds`：`[2000, 3000, 4000]`
- `data.drr_thresholds`：`[1000, 2000, 3000]`
- `train.detach_stage1_to_stage2`：若为 `true`，Stage2 不把梯度回传到 Stage1，更稳一些
- `val.save_inputs`：验证时是否保存 `KV`、`DRR`、`spine` 的 `nii.gz`

## 输出

- `outputs/experiment_name/checkpoints/`
  - `best.pt`
  - `last.pt`
- `outputs/experiment_name/val_epoch_XXX/`
  - `case_xxx_fake_drr.nii.gz`
  - `case_xxx_pred_spine.nii.gz`
  - 可选：`case_xxx_kv.nii.gz`、`case_xxx_drr.nii.gz`、`case_xxx_spine.nii.gz`
- `outputs/experiment_name/train.log`

## 说明

- 该工程默认按 **2D 单张图像** 处理（`H×W`）
- 若 `nii.gz` 是 `H×W×1` 或 `1×H×W` 也会自动压缩成 2D
- 如果你的 `nii.gz` 是真正的 3D 体数据（多层切片），需要额外修改数据集逻辑
- 验证输出保存时，会使用输入 `KV` 的 `affine/header`，并把网络输出乘回 `intensity_scale`

