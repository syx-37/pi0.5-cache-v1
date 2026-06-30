# pi0.5 接入 VLA-Cache 视觉 token 理论压缩统计执行文档

本文档对应第一版安全集成：只统计 VLA-Cache 对视觉 token 的理论压缩率，不做真实 KV 跳算，不修改模型输出。

也就是说，这一版用于回答：

```text
不接入 VLA-Cache 时，pi0.5 每步用了多少视觉 token？
按 VLA-Cache 静态视觉 token 选择逻辑，理论上可以复用/压缩多少？
压缩后的等效视觉 token 数是多少？
压缩率是多少？
成功率是否保持不变？
```

## 1. 当前版本做了什么

对 LIBERO 默认 pi0.5 输入，如果 `num_images_in_input = 2`，通常每个相机经过 SigLIP 后是 256 个视觉 token：

```text
2 个相机 * 256 token = 512 个视觉 token / inference step
```

本补丁不会写死 512，而是根据实际进入 `embed_prefix()` 的有效图像数量动态统计。如果服务器实际跑出来是 3 个相机，就会统计成约 768 个视觉 token。

统计字段含义：

```text
baseline_visual_tokens: 原始视觉 token 数
reused_visual_tokens: 理论上可复用的视觉 token 数
effective_visual_tokens: baseline_visual_tokens - reused_visual_tokens
visual_compression_rate: reused_visual_tokens / baseline_visual_tokens
```

`reused_visual_tokens` 来自相邻推理步视觉 token embedding 的 cosine similarity。默认阈值是：

```text
model.vla_cache_sim_threshold=0.996
```

## 2. 当前版本不做什么

这一版不做真实 KV cache 跳算，也不改 `past_key_values`。

原因是服务器当前 transformers `DynamicCache.update()` 只支持 append：

```python
self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
```

它没有 VLA-Cache 论文/OpenVLA patch 里需要的按视觉 token 位置覆盖、静态 token 复用、cache_position 控制等能力。强行做真实 KV 复用风险很高，所以第一版先做理论压缩率和成功率表格。

因此，开启本补丁后，成功率理论上应和 baseline 基本一致。如果成功率明显变化，优先检查代码接入位置或 episode/reset 状态。

## 3. 本地已经准备好的文件

本目录 `server_vla_cache_patch/` 现在是一套完整补丁文件：

```text
server_vla_cache_patch/
  __init__.py
  pi0_model.py
  vla_cache_state.py
  evaluate.py
  vla_cache_eval_table.py
  VLA_CACHE_STATIC_SELECTION_RUNBOOK.md
```

对应服务器目标位置：

```text
models/pi0/__init__.py
models/pi0/pi0_model.py
models/pi0/vla_cache_state.py
scripts/evaluate.py
scripts/vla_cache_eval_table.py
```

## 4. 上传到服务器

建议把整个 `server_vla_cache_patch` 文件夹上传到服务器项目根目录：

```bash
/media/SSD7/syx/projects/EAI-main/server_vla_cache_patch
```

服务器项目根目录是：

```bash
cd /media/SSD7/syx/projects/EAI-main
```

## 5. 替换前先备份服务器文件

在服务器上执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

mkdir -p backup_before_vla_cache_pi05

cp models/pi0/__init__.py backup_before_vla_cache_pi05/__init__.py
cp models/pi0/pi0_model.py backup_before_vla_cache_pi05/pi0_model.py
cp models/pi0/vla_cache_state.py backup_before_vla_cache_pi05/vla_cache_state.py
cp scripts/evaluate.py backup_before_vla_cache_pi05/evaluate.py
```

## 6. 替换服务器文件

在服务器上执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

cp server_vla_cache_patch/__init__.py models/pi0/__init__.py
cp server_vla_cache_patch/pi0_model.py models/pi0/pi0_model.py
cp server_vla_cache_patch/vla_cache_state.py models/pi0/vla_cache_state.py
cp server_vla_cache_patch/evaluate.py scripts/evaluate.py
cp server_vla_cache_patch/vla_cache_eval_table.py scripts/vla_cache_eval_table.py
```

## 7. 替换后先做语法检查

在服务器环境里执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

python -m py_compile \
  models/pi0/__init__.py \
  models/pi0/pi0_model.py \
  models/pi0/vla_cache_state.py \
  scripts/evaluate.py \
  scripts/vla_cache_eval_table.py
```

如果这里报错，先不要跑评估，把报错信息发给我。

## 8. Baseline 评估命令

每次测试：`libero_spatial` 的 10 个任务，每个任务 50 个 episode。

```bash
cd /media/SSD7/syx/projects/EAI-main

python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  model.use_vla_cache=false \
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_ppo_libero_spatial_10tasks_50eps_baseline \
  2>&1 | tee baseline_10tasks_50eps.log
```

## 9. VLA-Cache 理论压缩统计评估命令

这条命令会统计视觉 token 理论压缩率，但不改变模型输出：

```bash
cd /media/SSD7/syx/projects/EAI-main

python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  model.use_vla_cache=true \
  model.vla_cache_stage=static_selection \
  model.vla_cache_sim_threshold=0.996 \
  model.vla_cache_log_interval=50 \
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_ppo_libero_spatial_10tasks_50eps_vlacache_t0996 \
  2>&1 | tee vlacache_t0996_10tasks_50eps.log
```

如果你想测不同阈值，可以只改这一项：

```bash
model.vla_cache_sim_threshold=0.995
model.vla_cache_sim_threshold=0.997
model.vla_cache_sim_threshold=0.999
```

## 10. 生成压缩率和成功率表格

评估结束后执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

python scripts/vla_cache_eval_table.py \
  --baseline-log baseline_10tasks_50eps.log \
  --cache-log vlacache_t0996_10tasks_50eps.log \
  --csv-out vla_cache_libero_spatial_10tasks_50eps.csv
```

终端会打印 Markdown 表格，同时生成：

```text
vla_cache_libero_spatial_10tasks_50eps.csv
```

表格中重点看：

```text
success_rate
baseline_visual_tokens
reused_visual_tokens
effective_visual_tokens
visual_compression_rate
```

## 11. 预期结果

如果 LIBERO 输入确实是 2 个有效相机，单步 baseline 视觉 token 通常约为：

```text
512
```

如果统计结果显示单步约为：

```text
768
```

说明当前评估实际用了 3 个有效相机。

这不是错误，说明实际 server pipeline 和我们原先按 `num_images_in_input=2` 的假设不同，应以统计结果为准。

## 12. 出问题时怎么回退

如果替换后评估无法运行，可以先回退：

```bash
cd /media/SSD7/syx/projects/EAI-main

cp backup_before_vla_cache_pi05/__init__.py models/pi0/__init__.py
cp backup_before_vla_cache_pi05/pi0_model.py models/pi0/pi0_model.py
cp backup_before_vla_cache_pi05/vla_cache_state.py models/pi0/vla_cache_state.py
cp backup_before_vla_cache_pi05/evaluate.py scripts/evaluate.py
```

然后把报错日志、`py_compile` 输出、以及替换后的几个文件发给我，我再继续帮你定位。

## 13. 下一阶段怎么做真实 KV 跳算

如果第一阶段统计结果显示压缩率足够高，并且成功率没有下降，下一阶段才建议做真实 KV 复用。

真实 KV 跳算需要继续改：

```text
models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/*
```

重点是让视觉 token 的 K/V 能按位置复用，而不是让 `DynamicCache` 一直 append。这个阶段会改变推理路径，需要单独做 correctness 对齐测试。
