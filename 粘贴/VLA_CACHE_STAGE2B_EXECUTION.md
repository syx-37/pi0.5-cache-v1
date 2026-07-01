# pi0.5 VLA-Cache Stage 2B 执行文档

本文档对应 Stage 2B：在 Stage 2A 已经验证真实 KV 复用路径可用之后，进一步做视觉 token 的真实跳算。

服务器项目根目录：

```text
/media/SSD7/syx/projects/EAI-main
```

## 1. Stage 2B 目标

Stage 2A 已经做到：

```text
静态视觉 token 的 KV 可以保留上一帧
非静态 token 的 KV 覆盖写入当前帧
但 prefix forward 仍然完整计算所有 token
```

Stage 2B 要做到：

```text
每个 env 样本单独选择静态视觉 token
被选中的静态视觉 token 不再进入后续 Gemma decoder 层计算
这些 token 的 K/V 直接沿用上一帧 cache
非静态视觉 token 和 language token 继续计算并写回原始 cache_position
```

成功后应看到：

```text
real_skipped_visual_tokens_total > 0
real_kv_reused_tokens_total > 0
real_kv_mode = skip_tokens
```

速度是否明显提升取决于跳算比例、batch padding、GPU kernel 开销和环境仿真耗时。

## 2. 默认参数

本阶段默认使用：

```text
vla_cache_stage=real_kv
vla_cache_real_kv=true
vla_cache_skip_tokens=true
vla_cache_sim_threshold=0.996
vla_cache_pruning_layers=[2,6,9,11]
vla_cache_top_k_per_camera=150
```

说明：

```text
每个相机 256 个视觉 token
top_k_per_camera=150 表示每个 env、每个相机最多跳过 150 个高相似视觉 token
pruning_layers=[2,6,9,11] 当前实现使用其中最小层 2 作为开始跳算层
```

也就是说，第 0、1 层仍完整计算；从第 2 层开始，静态视觉 token 不再作为 query 进入后续层。

## 3. 修改文件

本阶段需要替换这些服务器文件：

```text
models/pi0/__init__.py
models/pi0/pi0_model.py
models/pi0/vla_cache_state.py
models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
```

本地对应文件：

```text
粘贴/__init__.py
粘贴/pi0_model.py
粘贴/vla_cache_state.py
粘贴/models_pytorch/gemma_pytorch.py
```

本阶段不改：

```text
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
```

## 4. 实现方式

### 4.1 每个 env 单独选择跳算 token

`vla_cache_state.py` 会比较当前帧和上一帧的 prefix embedding：

```text
similarity >= vla_cache_sim_threshold
```

然后在每个 env、每个 camera 范围内取 top-k：

```text
top_k_per_camera=150
```

得到 `skip_token_mask: [batch, prefix_len]`。

### 4.2 Gemma decoder 层裁剪 query token

`gemma_pytorch.py` 在第 `min(vla_cache_pruning_layers)` 层开始，把 `hidden_states` 从：

```text
[B, prefix_len, hidden_dim]
```

裁剪成：

```text
[B, max_kept_len, hidden_dim]
```

其中每个 batch 样本保留的 token 可以不同，短样本用 padding 对齐。

### 4.3 KV cache 仍保持完整 prefix 长度

虽然 query token 被裁剪，KV cache 的 key/value 维度仍保持完整 prefix 长度：

```text
[B, num_heads, prefix_len, head_dim]
```

被跳过的视觉 token 不写入当前帧 K/V，继续使用上一帧 K/V。

### 4.4 prefix_output 恢复完整长度

为了保护 `use_vlm_value=True` 的 value head，Gemma forward 结束后会把输出恢复为：

```text
[B, prefix_len, hidden_dim]
```

当前帧计算过的位置写入当前输出，被跳过的位置沿用上一帧 `prefix_output`。

## 5. 替换命令

在服务器项目根目录执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

cp 粘贴/__init__.py models/pi0/__init__.py
cp 粘贴/pi0_model.py models/pi0/pi0_model.py
cp 粘贴/vla_cache_state.py models/pi0/vla_cache_state.py
cp 粘贴/models_pytorch/gemma_pytorch.py models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
```

## 6. 语法检查

```bash
python -m py_compile \
  models/pi0/__init__.py \
  models/pi0/pi0_model.py \
  models/pi0/vla_cache_state.py \
  models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
```

如果语法检查失败，不要跑评估。

## 7. Smoke Test

先跑小规模确认不会崩：

```bash
python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  +model.use_vla_cache=true \
  +model.vla_cache_stage=real_kv \
  +model.vla_cache_real_kv=true \
  +model.vla_cache_skip_tokens=true \
  +model.vla_cache_sim_threshold=0.996 \
  +model.vla_cache_pruning_layers=[2,6,9,11] \
  +model.vla_cache_top_k_per_camera=150 \
  num_gpus=1 \
  env.total_num_envs=2 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=2 \
  output.experiment_name=pi05_stage2b_smoke_skip_tokens \
  2>&1 | tee stage2b_smoke_skip_tokens.log
```

检查日志中应出现：

```text
real_kv_mode: skip_tokens
real_skipped_visual_tokens_total > 0
real_kv_reused_tokens_total > 0
```

## 8. 10 tasks x 50 episodes

Smoke test 正常后，再跑完整评估：

```bash
python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  +model.use_vla_cache=true \
  +model.vla_cache_stage=real_kv \
  +model.vla_cache_real_kv=true \
  +model.vla_cache_skip_tokens=true \
  +model.vla_cache_sim_threshold=0.996 \
  +model.vla_cache_pruning_layers=[2,6,9,11] \
  +model.vla_cache_top_k_per_camera=150 \
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_libero_spatial_10tasks_50eps_stage2b_skip_tokens \
  2>&1 | tee stage2b_skip_tokens_10tasks_50eps.log
```

## 9. 成功标准

建议和你的已知结果对比：

```text
baseline: 97.2%, latency 约 439.8 ms
Stage 2A: 97.4%, latency 约 447.4 ms
```

Stage 2B 理想结果：

```text
Success Rate 不明显低于 baseline
real_skipped_visual_tokens_total > 0
real_kv_reuse_rate 明显大于 2A 的 0.1427
Inference Latency 开始低于 2A
```

如果成功率明显下降，优先调保守参数：

```text
+model.vla_cache_top_k_per_camera=100
+model.vla_cache_pruning_layers=[6,9]
+model.vla_cache_sim_threshold=0.997
```

如果速度没有提升但成功率稳定，说明跳算路径有效但实际 kernel/padding 收益不够，可以继续优化 batch packing 或更早层跳算。
