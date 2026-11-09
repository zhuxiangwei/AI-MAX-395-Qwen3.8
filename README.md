# AI-MAX-395-Qwen3.8 — Qwen3.8 本地推理部署工程

AMD Ryzen AI Max+ 395（Strix Halo）128 GB 统一内存推理机上的 **Qwen3.8 双后端推理底座**：

| 后端 | 引擎 | 模型 | 端口 |
|------|------|------|------|
| **Flash-Next Q6**（cache-first MoE） | Strata v0.2.11（ROCm 7.14.1 本地编译） | Qwen3.8-Flash-Next-UD-Q6_K_XL（169.2 GB） | 8080 |
| **Qwen3.8-27B Q8**（Dense VLM） | llama.cpp v0.9.0（HIP/gfx1151） | Qwen3.8-27B-UD-Q8_K_XL（30 GB）+ MTPQ4 草稿 | 8080 |

仓库内容：**推理服务（systemd unit + 启动器 + 配置）、模型部署要点、实测性能基线**。视频生成线（MiniMax H3 / FastH3）见姊妹仓库 [AI-MAX-395-MiniMax-H3](https://github.com/zhuxiangwei/AI-MAX-395-MiniMax-H3)。

> 已脱敏的公开副本：密钥为占位符，路径为示例。两后端与视频线共用统一内存/GTT，**互斥**，同一时刻只跑一个；统一内存下切换需重启整机才能彻底归还 GTT（实测切换前 GTT 48.5 GiB / 可用 20.4 GiB，重启后 GTT 1.3 GiB / 可用 121.3 GiB）。

**占位符约定**：

| 占位符 | 替换为 |
|--------|--------|
| `/home/user` | 实际部署用户的主目录（systemd **user** unit 里也可用 `%h`） |
| `LLAMA_API_KEY` 环境变量 | 自行生成的随机密钥，如 `export LLAMA_API_KEY=$(openssl rand -hex 16)` |
| `CHANGE_ME_TO_A_RANDOM_SECRET` | Strata 配置里的 `api_key`，换成随机密钥 |

---

## 1. 部署前提（推理底座）

- **APU**：AMD Ryzen AI Max+ 395 w/ Radeon 8060S（**gfx1151**），16C/32T；128 GB LPDDR5X 统一内存（CPU/GPU 共享），推理可用约 124 GB
- **GTT 124 GB**，GRUB 内核参数：
  `amd_iommu=off amdgpu.gttsize=126976 processor.max_cstate=1 ttm.pages_limit=32505856 ttm.page_pool_size=32505856 amdgpu.no_system_mem_limit=1`
- **ROCm 7.14.1**（`/opt/rocm/core-7.14`）

⚠️ **`ttm.pages_limit` 必须与 `amdgpu.gttsize` 对齐**（页数 = GiB × 262144）：对齐前 pages_limit=112G 曾卡住 gttsize 的 8 GiB——对齐后 Q6 才能落位。

---

## 2. 后端一：Flash-Next Q6 · Strata（cache-first）

### 2.1 部署终态

| 项 | 值 |
|---|---|
| 模型 | **Qwen3.8-Flash-Next**（125B 参数，另含 51B n-gram 嵌入 + 4B MTP 头 → 总存量 1800 亿；每 token 激活 60 亿；512 专家/48 层/262K 原生上下文）· **UD-Q6_K_XL**（6 分片 169.2 GB） |
| 版本 | Strata **v0.2.11 / engine 0.1.57**（源码 tag 本地 HIP/gfx1151 编译；官方无 Linux halo 预编译包） |
| 服务 | `strata.service` → `run-unsloth-ud-q6_k_xl.sh` → `strata-unsloth-ud-q6_k_xl.json` |
| 配置要点 | `--kv fp16`（原则不量化）· `--max-context 262144`（原则不缩减）· `--expert-cache 13000`（引擎扩 **15256 槽 / 63.23 GiB**，覆盖 62% 专家）· `--resident-budget-gib 43`（实锁 **35.0–35.4 GiB**，引擎按可用内存钳制，启动后运行中恒定）· `--mmap-experts`（冷尾 SSD 流式）· `--mtp mtp/rt`（**Q2_0 专家码 rt，唯一可用档**；官方 Q4_K_M/Q8_0/BF16 MTP 包因张量命名与块格式不匹配无法建 rt）+ `--spec 4 --spec-min-p 0.5` · `--vision` · `--ple-io direct`（PLE 表 54.4 GB 留 SSD）· `--expert-profile learned.bin` |
| 钉页 env | `HSA_USERPTR_FOR_PAGED_MEM=0`（禁内核回收映射页，防 verify 卡死） |
| 内存终态 | cache 63.2 + 驻留 35 + KV cells 3 + KV pinned 6 + vision/MTP/embd 3.1 + OS 5 ≈ **贴边稳定**（MemAvailable 0.4 GiB 零漂移，swap 吸收溢出） |
| 三层覆盖 | GPU cache 62% + RAM 驻留（画像排序的热尾）→ **~89% 专家零文件读取** |

**三层设计**：GPU cache（15256 热槽，画像预热，覆盖 62%）→ RAM 驻留（35 GiB 页锁定，CPU 池直算，覆盖至 ~89%）→ mmap/SSD（冷尾按需）。页锁定层内核不可驱逐 → 无主缺页风暴、无 verify 卡死。

> ⚠️ 上表内存是"启动初始终态"，不是运行中快照。运行中 MemAvailable 会被压到 0.6 GiB 量级（实测 0.62 GiB、swap 用 2.86 GiB、cgroup current 40.1 / peak 42.6 GiB），属正常。

> ❌ **不要加大 `--resident-budget-gib`**：43 的预算启动时就被钳制到实锁 35.42 GiB；decode 专家缓存命中已 98.1%，边际收益 <2%，而 MemAvailable 贴底、swap 已用，再锁只能从 swap 和 KV 页里挤，风险远大于收益。

### 2.2 实测性能（生产基线 757 请求 + 复核 533 请求）

口径：TG/PP 取逐任务中位（n_gen ≥ 100；PP 仅统计增量 ≥ 500 token 的真实 prefill）。会话最长 **230,064 tokens**。**10 小时 44 分连续运行零重启**，生成 512,005 tokens / 4.2 小时，MemAvailable 0.4 GiB 贴边稳定。

| ctx 段 | n | prefill 中位 | TG 中位 | MTP 接受 |
|---|---:|---:|---:|---:|
| 0-32k | 31 | 398.6 | 35.00 | 0.614 |
| 32-64k | 59 | 237.3 | 37.50 | 0.748 |
| 64-96k | 63 | 206.1 | 34.10 | 0.696 |
| 96-128k | 131 | 126.8 | 35.40 | 0.733 |
| 128-160k | 163 | 128.9 | 36.20 | 0.760 |
| 160-192k | 181 | 111.3 | 35.70 | 0.728 |
| 192-224k | 114 | 132.5 | 33.40 | 0.707 |
| 224-262k | 15 | 103.2 | 32.90 | 0.699 |

**总量**：757 任务（700 有效）· ctx 中位 **150,020** · prefill 中位 **163.6** t/s · **TG 中位 34.90** t/s（最好 54.7 / 最差 23.8）· MTP 接受中位 **0.727**。

**复核（533 请求）**：TG 中位 **36.90**（+5.7%）、P10/P90 = 32.30/44.90、冷读 PP 中位 484.1（≥20k 新读）、MTP 接受 0.693、decode 缓存命中 95.6%、KV 复用率 86.5%；TG 分档衰减 **−4.2%**（38.40 → 35.80）。日志零 Traceback / ERROR / OOM。

**成本几乎全在冷读**：16.5% 的请求吃掉 93.4% 的 prefill 读入；KV 复用率 86.5%，剩下全量重读单次 20 万 token 要 450–486 s。优化杠杆只有「别让前缀缓存失效」。

### 2.3 关键约束与已知坑

- ⚠️ **fp16 KV 与 262K 上下文是原则**：此前 Q6 的 3-17 分钟死亡循环 = 全量专家映射（106G）+ 裸 cache 双份占物理的超卖；终态靠 cache/驻留重分配 + 页锁定解决。
- ⚠️ **`--mtp` 接 mtp/rt 目录路径**（隐藏参数），裸 `--mtp` 会吞下一个参数。
- ⚠️ **MemAvailable 贴底（0.4 GiB）是设计终态**：勿在此状态再开其他吃内存进程（OOM killer 会挑最大的进程——就是引擎）。
- ⚠️ **不要在 prefill 中途 kill 服务**：GPU/KFD 状态搞脏后只能重启机器。
- ⚠️ **长任务突然变慢的主因是会话缓存没开**：启动即打 `conversation cache off: not enough free RAM to park conversations`。没有第二层存储时 arena 只装一条分支，**任何穿插进来的请求都会把主会话顶掉**，下一轮从 token 0 重读：160k 上下文实测 read 中位 **355 s**。穿插小请求时全冷率 76–100%，侧边调用（标题、补全建议等）要关掉或错开。
- ⚠️ **`reasoning_effort` 是注入 prompt 的一句话，不是采样参数**：`high` ≡ `xhigh`，`medium` = 无注入（模型原生行为）。本部署常态为 **`medium`** + `env_to_user: true`（把客户端 `<env>` 块从 system 开头挪到首条 user message，稳定前缀缓存）。
- ⚠️ 别把 `fit_max_tokens` / `answer_room` 当质量主因：实测主任务思考量 p99 仅 5442 tokens，这两条链路从未碰到过思考。真正控思考长度的是 `reasoning_budget_tokens`（32768）。
- 观测：`STRATA_REQUEST_LINES=1` 让引擎逐请求打 `prompt N tokens = R reused + K read`。
- ⚠️ **strata 只绑 `127.0.0.1:8080`**；`args_written[]` 是启动参数快照，判生效路径看 `args[]` 或 `/proc/<pid>/cmdline`。

---

## 3. 后端二：Qwen3.8-27B Q8 · llama.cpp

### 3.1 部署终态

| 项 | 值 |
|---|---|
| 模型 | `Qwen3.8-27B-UD-Q8_K_XL`（30 G）Dense 27B，原生 VLM + Thinking |
| 引擎 | llama.cpp **v0.9.0**（`build-hip714-v0.9.0`）+ ROCm 7.14.1 |
| 服务 | `llama-router.service` → `llama-router.sh`（router :8080，`--models-max 1`），模型参数在 `router-preset.ini` |
| MTPQ4 sidecar | `spec-draft-model = draft/mtp-Qwen3.8-27B-Q4_0.gguf`（1.3 G）+ `--spec-draft-n-max 3 --spec-draft-ngl 99` |
| KV | fp16 · `--cache-ram 49152`（48 GiB **主存 prompt cache**：整段会话可 park/restore，这是 27B 不怕被抢 arena 的原因）· `kv-unified=1` · `flash-attn=auto` |
| 采样 | temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence-penalty 0.0（官方推荐值） |
| 思考 | reasoning on / effort **xhigh** · `chat-template-kwargs {"preserve_thinking":true}` |
| 目录 | 主模型 `model/`、mmproj `mmproj/`（**必需资产**）、draft `draft/`——`models-dir` 里任何额外 GGUF 都会被注册成独立模型 |

> ⚠️ v0.9.0 把 server 主体拆到 `bin/libllama-server-impl.so`，`llama-server` 仅约 17 KB，`LD_LIBRARY_PATH` **必须包含 `bin/`**（脚本已带）。
>
> 📝 v0.9.0 的 `MTMD_VIDEO` 默认启用（`--video-fps`），视频输入（抽帧喂 mmproj）依赖 PATH 中的 ffmpeg；文本/图片推理不依赖。

### 3.2 实测性能（256K 等宽 8 档，1614 任务生产负载，含 MTPQ4 sidecar）

| ctx 段 | n | prefill 中位 | TG 中位 | MTP 接受 |
|---|---:|---:|---:|---:|
| 0-32k | 63 | 279.0 | 15.58 | 0.642 |
| 32-64k | 199 | 232.1 | 14.77 | 0.638 |
| 64-96k | 284 | 186.6 | 15.23 | 0.746 |
| 96-128k | 264 | 152.9 | 14.74 | 0.789 |
| 128-160k | 248 | 132.5 | 13.21 | 0.741 |
| 160-192k | 272 | 116.6 | 11.99 | 0.704 |
| 192-224k | 250 | 103.4 | 11.64 | 0.732 |
| 224-256k | 34 | 98.4 | 10.02 | 0.648 |

**总量**：1614 任务 · ctx 中位 **127,772** · prefill 中位 **166.4** t/s · **TG 中位 13.63** t/s · MTP 接受 **0.719**。TG 随 ctx 衰减 **−36%**（稠密物理极限）。

### 3.3 MTPQ4 sidecar 优势

MTPQ4 草稿（Q4_0 量化的独立草稿文件，全量上 GPU）较内置 MTP：**TG +8.4%~+43.4%**（32-96k 最大）、命中率 0.62~0.67 → 0.80。它是 24 格草稿方案矩阵中**唯一全档非负**的方案。机制教训：**草稿开销随 ctx 线性增长、收益不随 ctx 增长**，凡评估草稿方案必须按 ctx 分档测。

### 3.4 已验证的否决项（勿重试）

- 主模型 KV 量化 `--cache-type-k/v q8_0`：**−4.81%**（短 ctx dequant 开销 > 带宽节省）。
- `--spec-draft-p-min 0.75`：接受率 +44% 是分母缩小假象，吞吐 **−10.7%**。
- ngram 兜底：均值 +6% 全靠单极值，中位 −0.5% ⇒ 非无条件正收益。
- ROCm 10.0/10.1：中小 ctx +8~13% 但 **>96k 退化 −6~14%** ⇒ 维持 7.14.1。
- MTP `n-max 4→3`：160k+ **+46.6%**。

---

## 4. 后端对比（同口径 8 档实测）

> 口径完全对齐（n_gen ≥ 100；PP 仅增量 ≥ 500 token）。27B 侧为 sidecar 优化后的生产数据；Q6 侧含 MTP。

| ctx 段 | Q6 TG | 27B TG | **TG 倍数** | Q6 prefill | 27B prefill | **PP 倍数** |
|---|---:|---:|---:|---:|---:|---:|
| 0-32k | 35.00 | 15.58 | **2.25×** | 398.6 | 279.0 | 1.43× |
| 32-64k | 37.50 | 14.77 | **2.54×** | 237.3 | 232.1 | 1.02× |
| 64-96k | 34.10 | 15.23 | **2.24×** | 206.1 | 186.6 | 1.10× |
| 96-128k | 35.40 | 14.74 | **2.40×** | 126.8 | 152.9 | 0.83× |
| 128-160k | 36.20 | 13.21 | **2.74×** | 128.9 | 132.5 | 0.97× |
| 160-192k | 35.70 | 11.99 | **2.98×** | 111.3 | 116.6 | 0.95× |
| 192-224k | 33.40 | 11.64 | **2.87×** | 132.5 | 103.4 | 1.28× |
| 224-256k | 32.90 | 10.02 | **3.28×** | 103.2 | 98.4 | 1.05× |

**读法**：
1. **全档 2.24-3.28 倍**，且 27B 侧已是 sidecar 优化后的最有利口径。
2. **平坦度是本质优势**：27B TG 随 ctx 衰减 −36%；Q6 仅 −6%——cache-first 三层分层把 MoE 的上下文衰减几乎压平，上下文越深倍数越大。
3. **PP 持平**：总量 0.98×，浅层 Q6 反快 1.43×。
4. 512K tokens / 230K 深度 / 10 小时 44 分连续真实 agentic 负载：**零重启、零死亡、零 verify 超时**。

---

## 5. 采样与思考模式（模型配置）

| 模型 | 官方思考模式采样 |
|------|------------------|
| Qwen3.8-27B | `temp=1.0 / top_p=0.95 / top_k=20 / min_p=0 / pp=0.0` |
| Flash-Next | 同家族配方 |

- `reasoning = on` 注入 `enable_thinking`；`reasoning_effort` 仅 Qwen3.8 模板原生支持（xhigh/medium/low，默认 xhigh）；Strata 侧把 `none` 映射为 `enable_thinking=false`。
- ⚠️ **preserve 变量名陷阱**：llama.cpp 的 `--reasoning-preserve` 注入 `preserve_reasoning`，但 Qwen 模板读 **`preserve_thinking`** → 必须用 `--chat-template-kwargs {"preserve_thinking": true}`。
- **别关 `preserve_thinking`**：关掉会让 xhigh 退化成 medium 的表现且更费 token，同时破坏前缀缓存。切 effort 会改 system prompt 开头 → 整条 prefix cache 失效。27B 主力编程任务维持 `xhigh`。
- **模型命名：无别名，传全名**。`router-preset.ini` 不配 `alias`，`/v1/models` 返回的 `id` = preset 段名。Strata 不校验模型名（任意字符串接受）；llama.cpp 侧严格校验。

---

## 6. 引擎构建（ROCm / gfx1151）

### 6.1 llama.cpp

Vulkan vs ROCm 对比（27B Dense 长上下文）：192K 总耗时 Vulkan 116 分钟 vs **ROCm 28 分钟（4.1×）**，短上下文 prefill ROCm 快 1.2-1.8× ⇒ 只维护 HIP 构建。

**升级流程（固定）：先 pull 到官方主线，再 checkout 到 release tag，最后才编译。**

```bash
cd ~/llama.cpp
git -c http.version=HTTP/1.1 pull --ff-only --depth=1
git -c http.version=HTTP/1.1 fetch --depth=1 origin tag v0.9.0
git checkout --detach v0.9.0

export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
cmake -B build-hip714-v0.9.0 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_RPC=OFF \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=OFF -DGGML_HIP_EXPORT_METRICS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_IS_DEV=OFF
cmake --build build-hip714-v0.9.0 -j16
```

- ⚠️ **不要**下载 tarball 覆盖工作区再造本地 commit：commit sha 会与官方分叉并嵌进版本串。切换源码后**必须重新 cmake configure** 再增量编译。
- ⚠️ `LLAMA_BUILD_IS_DEV` 默认 ON → 版本串带 `-dev`；判版本看 `llama-server --version`。
- ⚠️ **删/换构建目录前必须先重启服务**（运行中进程靠 inode 存活，下一次 spawn 必败）。
- 以最新 release tag 为准，不追 main；编到版本化目录名；改 preset 前先看 release notes（v0.4.1 曾把 `mlock=1` 改成 `load-mode` 导致重启循环）。
- 运行时 env：`ROCBLAS_USE_HIPBLASLT=1`，`LD_LIBRARY_PATH` 含 `/opt/rocm/core-7.14/lib` 与构建 `bin/` 目录（启动脚本已带）。

### 6.2 Strata

官方无 Linux halo 预编译包：checkout 对应 tag 源码，本机 HIP/gfx1151 编译（`build-halo`），ggml 固定提交；升级后冒烟看 `hipBLASLt tuning enabled`、驻留 GiB、出 token。hipBLASLt tuning 表缓存于 `tools/hip/gfx1151-hipblaslt-*.txt`（配置里 `STRATA_HIPBLASLT_TUNING` 指向它）。

---

## 7. 服务安装与文件清单

unit 为 **user** 级（装到 `~/.config/systemd/user/`），脚本装到 `~/scripts/`：

| 文件 | 部署位置 | 用途 |
|------|---------|------|
| `strata.service` / `run-unsloth-ud-q6_k_xl.sh` / `strata-unsloth-ud-q6_k_xl.json` | `~/.config/systemd/user/` / `~/Strata/` | Flash-Next Q6 后端 |
| `llama-router.service` / `llama-router.sh` / `router-preset.ini` | `~/.config/systemd/user/` / `~/scripts/` / 模型目录 | 27B Q8 后端 |
| `rocm-wait-devices.sh` | `~/scripts/` | 两服务共用 ExecStartPre：开机后 `/dev/kfd`、`/dev/dri/renderD*` 由 udev 延迟数秒创建，user 级 systemd 不支持 device unit，显式轮询兜底 |

```bash
systemctl --user daemon-reload
systemctl --user start strata.service      # 或 llama-router.service（二选一）
curl -s http://127.0.0.1:8080/v1/models    # 401 = 活着
```

模型权重不入库（体积原因）：Flash-Next UD-Q6_K_XL 与 27B UD-Q8_K_XL 从 ModelScope / hf-mirror.com 获取；27B 另需 mmproj（F16）与 MTP Q4_0 草稿文件；Flash-Next 另需 mmproj（BF16）与 MTP rt（Q2_0）包。

---

*硬件：AMD Ryzen AI Max+ 395 · 128 GB 统一内存（GTT 124G）· ROCm 7.14.1 · Strata v0.2.11 + Flash-Next UD-Q6_K_XL / llama.cpp v0.9.0 + Qwen3.8-27B UD-Q8_K_XL*