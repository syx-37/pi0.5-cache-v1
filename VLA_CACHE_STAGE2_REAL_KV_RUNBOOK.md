# pi0.5 接入 VLA-Cache 真实 KV 复用/跳算执行文档

本文档是 Stage 2 执行文档。Stage 1 已经完成的是“理论压缩率统计”，没有改变模型输出，也没有真实跳过 KV 计算。

Stage 2 的目标是把 VLA-Cache 真正接入 pi0.5 的 prefix VLM 推理路径，让静态视觉 token 的 K/V 可以跨相邻环境步复用，并逐步做到少算一部分视觉 token。

## 1. 当前文件是否齐全

现在关键文件已经齐了。

服务器侧 pi0.5 文件：

```text
models/pi0/__init__.py
models/pi0/pi0_model.py
models/pi0/vla_cache_state.py
models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
models/pi0/openpi/openpi/models_pytorch/pi0_pytorch.py
models/pi0/openpi/openpi/models_pytorch/preprocessing_pytorch.py
```

服务器侧 transformers_replace 文件：

```text
models/pi0/openpi/openpi/models_pytorch/transformers_replace/__init__.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/__init__.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/__init__.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/paligemma/__init__.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/siglip/__init__.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/siglip/check.py
```

说明：服务器目录里没有本地 `configuration_paligemma.py`，这是正常的。当前 `modeling_paligemma.py` 使用的是：

```python
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig
```

也就是直接用 pip-installed transformers 里的 PaliGemma config，本地不需要额外 patch。

VLA-Cache 原始参考代码也已经齐：

```text
C:/Users/syx/Desktop/vla-cache
C:/Users/syx/Desktop/transformers-vla-cache-openvla
```

## 2. Stage 2 做完后算不算真正嵌入 VLA-Cache

要分两步看。

### Stage 2A：真实 KV 覆盖复用

如果做到：

```text
上一帧静态视觉 token 的 K/V 被保留
当前帧只覆盖更新非静态视觉 token 的 K/V
suffix/action expert 读取的是复用后的完整 prefix KV
```

这时可以说：VLA-Cache 已经接入 pi0.5 的真实推理路径。

但 Stage 2A 不一定明显加速，因为可能仍然会保守地计算较多 token，用于 correctness 对齐。

### Stage 2B：真实 token 跳算

如果进一步做到：

```text
静态视觉 token 不再进入部分 Gemma 层的 Q/K/V/MLP 计算
DynamicCache 用 cache_position 对非静态 token 做 index_copy_ 覆盖写入
静态 token 位置继续沿用上一帧 KV
```

这才是更接近论文 VLA-Cache 的加速阶段。

所以：Stage 2A 完成后算“真实嵌入”；Stage 2B 完成并验证后，才算“真实加速版嵌入”。

## 3. VLA-Cache 原项目的关键逻辑

OpenVLA-OFT 参考代码里，VLA-Cache 的流程是：

1. 用相邻图像 patch cosine similarity 找静态 patch。
2. 用上一帧 attention 去掉任务相关 patch。
3. 把剩下的静态 token index 写入模型 config。
4. 模型 forward 时跳过这些 token 的计算，并用 cache_position 覆盖更新 K/V。

对应参考代码：

```text
vla-cache/src/openvla-oft/experiments/robot/vla_cache_utils.py
vla-cache/src/openvla-oft/experiments/robot/openvla_utils.py
transformers-vla-cache-openvla/src/transformers/cache_utils.py
transformers-vla-cache-openvla/src/transformers/models/llama/modeling_llama.py
```

最关键的 cache 改动是 `DynamicCache.update()` 支持 `cache_position` 覆盖写入：

```python
self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
```

而服务器当前 transformers 的 `DynamicCache.update()` 只会 append：

```python
self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
```

所以 Stage 2 必须先解决“按位置覆盖写 KV”的问题。

## 4. pi0.5 和 OpenVLA 的关键差异

OpenVLA 是 autoregressive action token generation，VLA-Cache 原代码接在 `predict_action(..., past_key_values=prompt_cache)` 上。

服务器 pi0.5 是：

```text
图像 + 语言 prefix 先过 PaliGemma VLM
得到 prefix past_key_values
action suffix / denoising expert 多步读取这个 prefix KV
```

pi0.5 的关键路径在：

```text
models/pi0/pi0_model.py
models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
```

Stage 2 只能改 prefix VLM 部分，不能破坏后面的 suffix denoising。

## 5. 推荐实施路线

不要一口气直接做最终版跳算。推荐按三个小阶段推进。

### 5.1 Stage 2A-1：实现可覆盖写入的 cache 类

新增或扩展一个 pi0.5 本地 cache 类，例如：

```text
models/pi0/vla_cache_state.py
```

或者新建：

```text
models/pi0/vla_dynamic_cache.py
```

它需要支持：

```python
update(key_states, value_states, layer_idx, cache_kwargs)
```

并且当 `cache_kwargs["cache_position"]` 是多个位置时，用：

```python
index_copy_(2, cache_position, key_states)
index_copy_(2, cache_position, value_states)
```

当 `cache_position` 是单个 token 时，仍允许 append，用于兼容 suffix 或生成式路径。

注意：不要直接改 pip-installed transformers 包。优先在 `models/pi0` 内部实现局部 cache 类，然后让 pi0.5 prefix path 使用它。

### 5.2 Stage 2A-2：真实 KV 复用，不做 token 删除

第一轮真实 KV 测试先不要删 token，只做：

```text
当前 prefix 完整计算
对静态视觉 token 位置保留上一帧 K/V
对非静态 token 位置覆盖写入当前帧 K/V
保存当前 K/V 给下一步
下一步验证 cache_position 覆盖写入不会破坏 suffix denoising
```

这个阶段预期：

```text
成功率应与 baseline 基本一致
速度不一定提升
视觉 token 统计仍可输出
VLA_CACHE_EVAL_STATS_JSON 中 real_kv_reused_tokens_total 应该大于 0
```

如果这个阶段都不稳定，不能进入 token 跳算。

### 5.3 Stage 2B：真正删除静态视觉 token 的计算

在 prefix VLM forward 中，针对静态视觉 token：

```text
hidden_states 删除静态 token
position_ids 改成剩余 token 的原始位置
cache_position 改成剩余 token 的原始位置
attention_mask 的 query 维度同步裁剪
key/value cache 的 key 维度保持完整 prefix 长度
```

当前 token 的新 K/V 写回非静态位置；静态位置保留上一帧 K/V。

参考 OpenVLA 逻辑：

```python
mask = ~torch.isin(cache_position, selected_reusable_patches)
new_cache_position = cache_position[mask]
hidden_states = hidden_states[..., mask, :]
position_ids = new_cache_position.unsqueeze(0)
cache_position = new_cache_position
```

pi0.5 里要把这段逻辑移植到 Gemma prefix 层循环中。

## 6. 需要修改的文件

Stage 2 预计需要修改：

```text
models/pi0/vla_cache_state.py
models/pi0/pi0_model.py
models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
```

可能需要修改：

```text
models/pi0/__init__.py
scripts/evaluate.py
scripts/vla_cache_eval_table.py
```

一般不需要修改：

```text
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py
models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py
```

PaliGemma 和 SigLIP 更多是提供图像 embedding，不是 KV 跳算核心。

## 7. 需要新增的配置项

建议保留 Stage 1 配置，并新增：

```text
model.vla_cache_stage=real_kv
model.vla_cache_real_kv=true
model.vla_cache_skip_tokens=false
model.vla_cache_pruning_layers=[2,6,9,11]
model.vla_cache_top_k_per_camera=150
model.vla_cache_task_relevant_top_k=100
model.vla_cache_reset_on_episode=true
model.vla_cache_debug_compare=false
```

推荐阶段：

```text
real_kv + skip_tokens=false：先验证真实 KV 覆盖
real_kv + skip_tokens=true：再验证真实跳算
```

## 8. 重要正确性约束

### 8.1 episode 边界必须 reset

每个 episode 开始时必须清空：

```text
prev_prefix_embs
prev_visual_token_mask
prev_past_key_values
prev_images
selected_reusable_indices
```

否则会把上一个 episode 的视觉 KV 错复用到下一个 episode。

### 8.2 batch 内环境结束要谨慎

你现在 `env.total_num_envs=10`，batch 里 10 个环境可能不是同时结束。理想做法是按 env id 单独 reset cache。

如果当前 evaluator 没有暴露 per-env done/reset 信息，Stage 2 第一版建议更保守：

```text
每个 evaluation round 或 episode boundary 全 batch reset
```

这样损失一些可复用机会，但更安全。

### 8.3 use_vlm_value 要注意

`pi0_model.py` 里有：

```python
if self.use_vlm_value:
    values_vlm = self.get_value_from_vlm(prefix_output)
```

如果真实跳算后 `prefix_output` 只剩非静态 token，那么 VLM value head 可能不再等价。

所以 Stage 2B 第一版建议加保护：

```text
如果 use_vlm_value=True，则禁止 skip_tokens=true
```

或者仅允许 Stage 2A，不删除 token。

### 8.4 suffix expert 读取的 KV 长度必须完整

虽然 prefix 里跳过了一些视觉 token 的计算，但传给 suffix expert 的 `past_key_values` 必须仍然包含完整 prefix 序列长度。

也就是说：

```text
cache.key_cache[layer].shape[-2] == 原始 prefix 长度
cache.value_cache[layer].shape[-2] == 原始 prefix 长度
```

如果 KV 长度变短，suffix attention mask 会错位。

## 9. 服务器替换前备份

在服务器执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

mkdir -p backup_before_vla_cache_stage2

cp models/pi0/__init__.py backup_before_vla_cache_stage2/__init__.py
cp models/pi0/pi0_model.py backup_before_vla_cache_stage2/pi0_model.py
cp models/pi0/vla_cache_state.py backup_before_vla_cache_stage2/vla_cache_state.py
cp models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py backup_before_vla_cache_stage2/gemma_pytorch.py
cp models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py backup_before_vla_cache_stage2/modeling_gemma.py
```

## 10. Stage 2 代码替换后的语法检查

本地补丁文件对应服务器路径如下：

```text
server_vla_cache_patch/pi0_model.py
  -> models/pi0/pi0_model.py

server_vla_cache_patch/__init__.py
  -> models/pi0/__init__.py

server_vla_cache_patch/vla_cache_state.py
  -> models/pi0/vla_cache_state.py

server_vla_cache_patch/gemma_pytorch.py
  -> models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
```

如果你在服务器上把补丁文件放在项目根目录的 `server_vla_cache_patch/`，可以这样替换：

```bash
cp server_vla_cache_patch/__init__.py models/pi0/__init__.py
cp server_vla_cache_patch/pi0_model.py models/pi0/pi0_model.py
cp server_vla_cache_patch/vla_cache_state.py models/pi0/vla_cache_state.py
cp server_vla_cache_patch/gemma_pytorch.py models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
```

替换 Stage 2 文件后执行：

```bash
cd /media/SSD7/syx/projects/EAI-main

python -m py_compile \
  models/pi0/__init__.py \
  models/pi0/pi0_model.py \
  models/pi0/vla_cache_state.py \
  models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py \
  scripts/evaluate.py
```

如果这里报错，不要跑评估，先修语法。

再跑一个很小的 cache 行为测试，确认静态位置保留、非静态位置覆盖：

```bash
python - <<'PY'
import torch
from models.pi0.vla_cache_state import VLAOverwriteDynamicCache

prev = VLAOverwriteDynamicCache()
k = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
prev.update(k, k + 100, 0, {"cache_position": torch.arange(5)})

cur = VLAOverwriteDynamicCache.from_cache(
    prev,
    reusable_token_positions=torch.tensor([1, 3]),
)
out_k, _ = cur.update(
    torch.full((1, 1, 5, 1), 9.0),
    torch.full((1, 1, 5, 1), 19.0),
    0,
    {"cache_position": torch.arange(5)},
)

print(out_k.flatten().tolist())
print(cur.vla_cache_last_update_stats)
assert out_k.flatten().tolist() == [9.0, 1.0, 9.0, 3.0, 9.0]
assert cur.vla_cache_last_update_stats["reusable_token_positions"] == 2
PY
```

## 11. Stage 2A 正确性测试命令

先跑小规模，确认不会崩：

```bash
python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  +model.use_vla_cache=true \
  +model.vla_cache_stage=real_kv \
  +model.vla_cache_real_kv=true \
  +model.vla_cache_skip_tokens=false \
  +model.vla_cache_sim_threshold=0.996 \
  num_gpus=1 \
  env.total_num_envs=2 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=2 \
  output.experiment_name=pi05_stage2a_smoke_real_kv_no_skip \
  2>&1 | tee stage2a_smoke_real_kv_no_skip.log
```

这个阶段的预期：

```text
不一定加速
不应该明显降成功率
应该能输出 VLA_CACHE_EVAL_STATS_JSON
real_kv_reused_tokens 或类似字段应该大于 0
```

## 12. Stage 2B 小规模跳算测试命令

Stage 2A 通过后，再开真实跳算：

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
  num_gpus=1 \
  env.total_num_envs=2 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=2 \
  output.experiment_name=pi05_stage2b_smoke_real_kv_skip \
  2>&1 | tee stage2b_smoke_real_kv_skip.log
```

这个阶段重点看：

```text
是否运行不崩
success rate 是否异常下降
inference latency 是否开始下降
effective_visual_tokens 是否下降
real_skipped_visual_tokens 是否大于 0
```

## 13. 10 任务 50 episode 正式评估命令

小规模通过后，再跑正式测试。

### baseline

```bash
python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  +model.use_vla_cache=false \
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_ppo_libero_spatial_10tasks_50eps_baseline_stage2 \
  2>&1 | tee baseline_stage2_10tasks_50eps.log
```

### Stage 2A：真实 KV 复用，不跳 token

```bash
python scripts/evaluate.py \
  --config-name=pi05_eval_libero \
  model.model_path=/media/SSD7/syx/projects/EAI-main/RLinf-Pi05-PPO-LIBERO-130 \
  +model.use_vla_cache=true \
  +model.vla_cache_stage=real_kv \
  +model.vla_cache_real_kv=true \
  +model.vla_cache_skip_tokens=false \
  +model.vla_cache_sim_threshold=0.996 \
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_libero_spatial_10tasks_50eps_stage2a_real_kv_no_skip \
  2>&1 | tee stage2a_real_kv_no_skip_10tasks_50eps.log
```

### Stage 2B：真实 KV 跳算

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
  num_gpus=1 \
  env.total_num_envs=10 \
  env.task_suite=libero_spatial \
  env.episodes_per_task=50 \
  output.experiment_name=pi05_libero_spatial_10tasks_50eps_stage2b_real_kv_skip \
  2>&1 | tee stage2b_real_kv_skip_10tasks_50eps.log
```

## 14. 表格统计

Stage 2 的统计脚本应扩展输出这些字段：

```text
success_rate
inference_latency_mean_ms
baseline_visual_tokens_total
reused_visual_tokens_total
effective_visual_tokens_total
theoretical_visual_compression_rate
real_skipped_visual_tokens_total
real_skip_rate
kv_cache_reuse_rate
gpu_peak_memory_gb
```

建议表格：

```text
baseline
stage1_static_selection_stats
stage2a_real_kv_no_skip
stage2b_real_kv_skip
```

## 15. 通过标准

Stage 2A 通过标准：

```text
py_compile 通过
小规模 smoke 通过
10tasks_50eps 成功率没有明显下降
VLA_CACHE_EVAL_STATS_JSON 有真实 KV 复用字段
```

Stage 2B 通过标准：

```text
成功率相比 baseline 下降不超过 1-2 个百分点
inference latency 下降
real_skipped_visual_tokens_total > 0
KV 长度始终等于完整 prefix 长度
没有 episode 跨界 cache 污染
```

如果 Stage 2B 成功率明显下降，应优先调高：

```text
model.vla_cache_sim_threshold=0.997
model.vla_cache_sim_threshold=0.999
```

或者减少跳算层：

```text
model.vla_cache_pruning_layers=[6,9]
```

## 16. 回退命令

如果 Stage 2 出问题，回退：

```bash
cd /media/SSD7/syx/projects/EAI-main

cp backup_before_vla_cache_stage2/__init__.py models/pi0/__init__.py
cp backup_before_vla_cache_stage2/pi0_model.py models/pi0/pi0_model.py
cp backup_before_vla_cache_stage2/vla_cache_state.py models/pi0/vla_cache_state.py
cp backup_before_vla_cache_stage2/gemma_pytorch.py models/pi0/openpi/openpi/models_pytorch/gemma_pytorch.py
cp backup_before_vla_cache_stage2/modeling_gemma.py models/pi0/openpi/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
```

## 17. 结论

当前文件已经足够进入 Stage 2。

下一步不是直接跑命令，而是先按本文档实现 Stage 2 代码补丁。补丁完成并通过 Stage 2A 后，可以说 VLA-Cache 已经真实接入 pi0.5 的 KV 路径；Stage 2B 通过后，才是带真实视觉 token 跳算加速的 VLA-Cache。
