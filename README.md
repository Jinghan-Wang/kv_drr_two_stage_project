# KV -> DRR -> spine_only (PyTorch Two-Stage Project)

这个工程实现了一个**总网络（一个 pth）**，内部包含两个阶段：

- Stage1: `KV(4ch) -> fake_DRR(1ch)`
- Stage2: `fake_DRR(4ch, 由 fake_DRR 自动阈值处理得到) -> pred_spine(1ch)`

其中：

- `KV` 输入 4 通道：`[原始, 2000-Max, 3000-Max, 4000-Max]`
- `DRR` 输入 4 通道：`[原始, 1000-Max, 2000-Max, 3000-Max]`
- `spine_only` 为 1 通道标签

训练时：

- 用 `KV` 经过 Stage1 生成 `fake_DRR`
- Stage1 用真实 `DRR` 监督
- `fake_DRR` 再自动做 `1000/2000/3000-Max` 阈值拼成 4 通道喂给 Stage2
- Stage2 用 `spine_only` 监督

验证时自动保存图像：

- 原始 KV
- 真实 DRR
- fake_DRR
- 真实 spine_only
- pred_spine

并打印 / 保存 loss。

---

## 1. 目录结构要求

推荐如下目录：

```text
your_dataset/
  train/
    kv/
      0001.png
      0002.png
      ...
    drr/
      0001.png
      0002.png
      ...
    spine/
      0001.png
      0002.png
      ...
  val/
    kv/
    drr/
    spine/
```

要求同名文件一一对应。

支持文件格式：

- `.png`
- `.jpg` / `.jpeg`
- `.bmp`
- `.tif` / `.tiff`
- `.npy`

> 如果是 16-bit 医学图像，建议尽量用 `.png/.tif/.npy` 保留灰度范围。

---

## 2. 安装依赖

```bash
pip install torch torchvision pillow numpy tqdm
```

---

## 3. 训练示例

```bash
python train.py \
  --data_root /path/to/your_dataset \
  --save_dir ./runs/exp1 \
  --epochs 200 \
  --batch_size 4 \
  --lr 1e-4 \
  --img_size 512 512 \
  --max_intensity 4095 \
  --lambda_drr_l1 1.0 \
  --lambda_drr_ssim 0.5 \
  --lambda_drr_grad 0.5 \
  --lambda_spine_l1 1.0 \
  --lambda_spine_ssim 0.5 \
  --lambda_spine_grad 0.5 \
  --detach_stage1_to_stage2 0
```

说明：

- `detach_stage1_to_stage2=0`：Stage2 的 loss 会反传到 Stage1
- `detach_stage1_to_stage2=1`：Stage2 的 loss 不反传到 Stage1（更稳，适合先训桥接）

---

## 4. 推理示例（测试时只有 KV）

```bash
python infer.py \
  --checkpoint ./runs/exp1/checkpoints/best.pth \
  --input_dir /path/to/test_kv \
  --output_dir ./runs/exp1/test_results \
  --img_size 512 512 \
  --max_intensity 4095
```

输出：

- `fake_DRR`
- `pred_spine`
- 可视化图

---

## 5. 工程文件说明

- `train.py`：训练脚本
- `infer.py`：测试脚本（只有 KV）
- `dataset.py`：数据加载及多阈值预处理
- `models.py`：UNet + TwoStageNet
- `losses.py`：L1 / SSIM / 梯度损失
- `utils.py`：保存图像、日志、随机种子等

---

## 6. 关于阈值图（2000-Max 等）

这里采用最常见的实现：

- 小于阈值的像素置零
- 大于等于阈值的像素保留原值
- 然后统一除以 `max_intensity` 归一化到 `[0,1]`

例如：

- `KV_2000 = (KV >= 2000) * KV`
- `DRR_1000 = (DRR >= 1000) * DRR`

如果你的前处理定义与此不同，可以在 `dataset.py` 和 `models.py` 中修改 `threshold_to_max()`。

---

## 7. 建议训练策略

如果你担心桥接不稳，可以按下面顺序：

1. 先训 Stage1 (`KV -> DRR`)
2. 先训 Stage2 (`DRR -> spine_only`)
3. 再训练总网络
4. 初期可设置 `--detach_stage1_to_stage2 1`
5. 稳定后可改为 `0` 微调

---

## 8. 验证图像保存

每个 epoch 验证时，会自动保存若干样本的对比图，例如：

- `KV`
- `DRR_gt`
- `DRR_pred`
- `spine_gt`
- `spine_pred`

帮助观察桥接是否成功、第二阶段是否稳定提取脊柱。

