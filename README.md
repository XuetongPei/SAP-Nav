# VLFM on LangMap — reproducible single-goal evaluation

官方 VLFM 全局策略，加 Habitat-Sim shortest-path local policy，在 LangMap 上评测。
支持 `object`、`room`、`region`、`instance` 四个 single-goal level，命令行自由选择。
默认运行 `object room`。本仓库提供从零启动的入口，不依赖历史运行目录、旧进程或聊天汇报服务。

这是 VLFM 的 LangMap 适配，不是 VLFM 官方发布的 LangMap 基线。
官方 VLFM commit 为 `584ed56008754fde7997d904983607def8328322`。
文本适配来自 ActiveSpin 的 `vlfm` 分支；navmesh 参数来自 LangMap。

## 实验设置与已有结果

| 项目 | 设置 |
|---|---|
| 数据 | LangMap `val`，HM3D 36个场景 |
| 样本数 | object 1460；room 2289；region 2687；instance 2878 |
| 全局策略 | VLFM `HabitatITMPolicyV2` / `ITMPolicyV2` |
| Local policy | Habitat-Sim GreedyGeodesicFollower；不加载 PointNav 网络 |
| 感知 | GroundingDINO SwinB + SAM2.1 Hiera-L + BLIP-2 ITC + YOLOv7-E6E |
| 数值精度 | DINO/SAM2.1 FP32；YOLO FP16；BLIP-2使用LAVIS原始精度配置 |
| 成功条件 | 终点到最近有效目标 viewpoint 的测地距离 **严格小于0.25 m** |
| SPL | 使用重建 navmesh 的起点距离和实际移动路径长度 |
| 步数 / 种子 | 每个episode最多500个action；base seed `20261019` |
| 机器人 | 半径0.18 m，高度0.88 m；相机高度0.88 m |
| RGB-D | 640×480，HFOV 79°；depth 0.5–5 m |
| 动作 | 前进0.25 m，转向30°，不允许sliding |
| Navmesh | LangMap Stretch流程，仅覆盖agent半径/高度；cell size/height均0.05 m |
| 到达半径 | 局部waypoint 0.25 m；VLFM物体停止半径0.9 m |
| 默认并发 | 一张A800；2套模型，每套16个agent，共32个 |
| 线程 | 每个DINO服务4线程，其余服务和policy进程1线程 |

本地历史全量结果如下。Region和instance仅提供运行支持，没有完整参考指标。

| Level | Episodes | SR | SPL |
|---|---:|---:|---:|
| Object | 1460 | 14.93% | 8.83% |
| Room | 2289 | 11.58% | 7.01% |

原运行在A800 80GB、14核CPU配额和120 GiB内存配额下进行。Object/room交叠运行，整个任务约12小时33分，包含暂停、线程测试及部分region工作；这不是新环境的耗时保证。
本仓库直接使用4线程DINO，无需重放当时的暂停/切换过程。
详细协议、固定episode ID和参考指标位于 [`reference/`](reference/)。

## 1. 安装环境

目标平台为 Linux x86_64、NVIDIA GPU，原实验为Ampere A800。
需要 Conda、Git、C/C++编译器和CUDA toolkit（原DINO构建使用11.8）。
系统需要OpenGL/EGL及X11运行库；Ubuntu可安装：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libgl1 libegl1 libglib2.0-0 libxcb1 libx11-6 libxcursor1 libxi6 libxinerama1 libxrandr2
cd /path/to/vlfm-langmap-repro
export CUDA_HOME=/usr/local/cuda-11.8   # 按本机安装位置设置
bash scripts/install_envs.sh
source scripts/env.sh
"$VLFM_PYTHON" -m scripts.download_models --bert
```

安装脚本创建三个独立环境，避免旧VLFM与新SAM2的Torch版本互相覆盖：

| 环境 | Python | 核心依赖 |
|---|---|---|
| `runtime/envs/vlfm` | 3.9 | Torch 1.12.1+cu113；LAVIS 1.0.2；旧Habitat导入依赖 |
| `runtime/envs/sim` | 3.11 | Habitat-Sim 0.3.3，原实验使用的固定wheel及SHA-256 |
| `runtime/envs/sam2` | 3.11 | Torch 2.10.0+cu128；SAM2固定commit |

安装依赖较大；外部源码下载到 `external/`，权重、缓存和环境放在 `runtime/`，均已加入 `.gitignore`。
权重按 [`configs/models.json`](configs/models.json) 校验SHA-256，BERT也固定revision。
推理服务离线读取已准备的模型缓存。模型、数据及依赖代码各自的许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

如已有相同版本环境，可在source之前设置 `VLFM_PYTHON`、`LANGMAP_PYTHON`、`SAM2_PYTHON`，无需重装。
可覆盖 `VLFM_RUNTIME`、`VLFM_MODELS`、`HF_HOME`、`TORCH_HOME`。
外部源码路径可以通过 `VLFM_DINO_SOURCE`、`VLFM_YOLO_SOURCE`、`VLFM_SAM2_SOURCE` 覆盖。
`scripts/env.sh` 根据自身位置确定仓库路径，不含个人工作区的绝对路径。

## 2. 配置数据

从 [LangMap](https://github.com/bo-miao/LangMap) 获取对应标注，并按 [HM3D](https://aihabitat.org/datasets/hm3d/) 的流程获取场景。
本仓库不重新分发标注、场景或模型权重。

```text
$EMBODIEDSCORE_DATA_ROOT/
└── langmap/annotations/<scene-id>.json       # 也支持.json.gz
$EMBODIEDSCORE_SCENE_ROOT/
└── hm3d/val/<scene-id>/<scene-short-id>.basis.glb
```

例如 `00800-TEEsavR23oF/TEEsavR23oF.basis.glb`。

```bash
export EMBODIEDSCORE_DATA_ROOT=/path/to/datasets
export EMBODIEDSCORE_SCENE_ROOT=/path/to/scenes
source scripts/env.sh
"$LANGMAP_PYTHON" -m scripts.check_data
"$LANGMAP_PYTHON" -m scripts.run --preflight --levels object room region instance \
  --output "$VLFM_RUNTIME/runs/four-level-preflight"
```

数据预检验证36个标注文件的内容哈希、黑名单过滤后的episode ID及顺序、场景文件存在性。
使用仓库内冻结的EmbodiedScore-envs加载器，不依赖其他私有工作区。
调整JSON空白/键顺序或压缩方式不会影响标注内容哈希。

## 3. 运行

先查看部署计划；此命令不加载数据和模型：

```bash
python -m scripts.run --plan --levels object room --workers 32 --replicas 2
```

实际运行前可单独重建36个场景的navmesh；运行时也会按需自动重建：

```bash
"$LANGMAP_PYTHON" -m scripts.prepare_langmap_navmeshes
```

默认复现object和room：

```bash
"$LANGMAP_PYTHON" -m scripts.run --gpu 0 --levels object room \
  --workers 32 --replicas 2 --output "$VLFM_RUNTIME/runs/object-room"
```

单独运行任意level，或者运行全部四层：

```bash
"$LANGMAP_PYTHON" -m scripts.run --levels region --output "$VLFM_RUNTIME/runs/region"
"$LANGMAP_PYTHON" -m scripts.run --levels instance --output "$VLFM_RUNTIME/runs/instance"
"$LANGMAP_PYTHON" -m scripts.run --levels object room region instance \
  --output "$VLFM_RUNTIME/runs/all-four-levels"
```

启动器从零创建每套的4个模型服务，分配agent，并在结束/异常/中断时关闭本次创建的进程。
`--workers` 必须能被 `--replicas` 整除。`--base-port` 默认为12181，每套占用4个端口，相邻套间隔10。
端口已占用时会直接报错，不接管已有服务。

在GPU上做短流程验证：

```bash
"$LANGMAP_PYTHON" -m scripts.run --levels object room region instance \
  --workers 4 --replicas 1 --limit-per-level 1 --max-actions-per-goal 8 \
  --output "$VLFM_RUNTIME/runs/smoke"
```

短流程会在manifest/summary中标记 `smoke_test=true`，不可当作全量指标。
重复相同命令和输出目录即可续跑，已完成episode会跳过。
配置、选样、标注或执行代码改变时必须使用新输出目录，防止混入不同协议结果。

## 4. 查看结果与测试

```bash
"$LANGMAP_PYTHON" -m scripts.summarize "$VLFM_RUNTIME/runs/object-room"
"$LANGMAP_PYTHON" -m pytest -q tests/test_runner.py test_langmap
"$VLFM_PYTHON" -m pytest -q tests/test_server_wrapper.py
```

可选的GitHub Actions配置保存在 `docs/github-actions-checks.yml.example`，当前未启用。
如需自动CI，可在具有workflow写入权限的情况下将其放到 `.github/workflows/checks.yml`。

| 产物 | 含义 |
|---|---|
| `manifest.json` / `episode-manifest.json` | 本次实际协议及选择的episode |
| `code-sha256.json` | 本地执行代码和配置的指纹 |
| `deployment.json` | 并发、线程、进程状态及起止时间 |
| `episodes/*.json` | 每个episode的SR/SPL、耗时、起终点位置及目标距离 |
| `trajectories/*.json` | 每一步动作、位姿、碰撞和局部控制状态 |
| `errors/*.json` | 基础设施异常，不计为导航失败 |
| `summary.json` | 已完成样本的聚合指标与完整性状态 |
| `services/`、`worker-*.log`、`policy-*.log` | 模型及agent日志 |

SR/SPL的分母是已完成episode；只有 `complete=true` 且 `smoke_test=false` 才表示所选level的全量评测完成。
每个episode的 `seconds` 包含仿真、推理和服务等待；并行episode的耗时相加不是GPU墙钟时间。

## 源码组织与验证范围

- `vlfm/`：官方策略、地图和视觉接口；本地算法改动与原实验保持一致。
- `scripts/`：可配置启动、评测、模型下载、依赖安装和汇总。
- `configs/`：当前部署默认值、固定依赖和模型校验信息。
- `config/`：保留的官方VLFM配置模板，实际LangMap协议见 `reference/protocol.json`。
- `third_party/`：本次所需的适配函数和冻结环境包，附来源、许可证和校验值。
- `reference/`：固定样本清单、标注哈希、协议与历史object/room全量指标。

复现包的整理验证记录见 [`reference/release-validation.json`](reference/release-validation.json)。
硬件或底层数值库变化可能影响浮点计算；参考分数不承诺逐位一致。
