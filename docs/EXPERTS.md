# 外部语音专家接入

## 交付状态与用途

本项目提供一个 CFPRF 原生帧分数导出接口，以及一个通用的 NumPy / CSV 帧分数导入接口。代码不包含外部模型源码、训练权重、数据集或代替真实专家输出的分数。外部仓库及权重需要使用者按作者说明自行准备。

**已验证的是分数方向、时间映射、输入检查、监督字段隔离与导入流程。当前交付环境没有执行真实 CFPRF checkpoint 推理，因此不提供任何真实数据上的检测性能结论。** 单元测试中的小数组只是接口测试数据。

第一轮建议只接 CFPRF，把音频、原生帧证据、本体结构和评测流程跑通。单专家不构成多模型融合创新；下一步应接入 PartialSpoof 或另一个有实际预测差异的频谱 / SSL 专家，再比较有无关系、条件可靠性与依赖建模的贡献。对同一模型改名、重复导出、换一个文件格式，不能产生新的互补证据。同一 backbone 的多尺度输出也应保留共享来源与依赖关系。

## 1. 共用数据约定

### Manifest 与标签隔离

Manifest 为 JSONL，至少包含以下字段：

```json
{"recording_id":"example_001","audio_path":"audio/example_001.wav","duration_sec":1.0,"sample_rate":16000,"split":"test"}
```

适配器只取 `recording_id`、`audio_path`、`duration_sec`、`sample_rate`、`split`。`label`、`fake_intervals` 等监督与审计字段可以存在，但不会进入模型调用、分数变换、时间长度推断或来源标识。相对音频路径在 CFPRF 导出时以 manifest 所在目录为基准解析。通用分数导入器不重新读取音频，只核对 manifest 中的推理元数据。

输出每一行为一个原生局部观测：

```json
{"observation_id":"obs_...","recording_id":"example_001","detector_id":"cfprf_fdn","run_id":"cfprf-fdn-v1","start_sec":0.0,"end_sec":0.02,"score":0.37,"status":"valid","provenance_id":"<本帧测量的 SHA-256>","quality":{}}
```

所有输出均统一为 **越大越偏向伪造**。`score_type` 保留真实语义：

| 类型 | 意义 | 如何转成伪造方向 |
|---|---|---|
| `margin` | 两类原始判别值之差、余弦判别值之差等；不承诺概率解释 | 两列取 fake − bona fide；单列若越大越真则取负 |
| `logit` | 上游确有该语义的二元 logit，或 softmax 之前两类 logits 的差 | 同上；不能仅因为数值是实数就称其为 logit |
| `probability` | 已是 [0, 1] 的概率 | 两列选 fake 概率；单列若是真实概率则取 1 − p |

适配器不对 margin 擅自套 sigmoid，也不把概率当作已充分校准。分数标准化和校准由主框架只使用对应训练 / 校准切分拟合。

Registry 的必需字段如下，适配器会补充来源与时间映射元数据：

```json
{"detectors":[{"detector_id":"cfprf_fdn","family":"xlsr300m","score_type":"margin","higher_is_fake":true,"scope":"local","checkpoint_sha256":"<SHA-256>","synthetic":false}]}
```

`family` 用来记录共享编码器或来源。它不是一个保证独立性的声明。该值应真实反映专家之间的依赖关系。

### 来源记录与错误处理

CFPRF 的输入来源包含音频、FDN checkpoint、XLS-R checkpoint、外部 FDN 源文件的 SHA-256，以及预处理方式和分数语义。通用导入的输入来源包含分数文件 SHA-256、checkpoint SHA-256 和明确的转换参数。在此基础上，再将录音、专家、原生帧序号、起止时间一起哈希，生成每一帧测量各自的 `provenance_id`。它标识单个输出测量，不能直接使用整条录音、整个 run 或整个分数文件的哈希来代替。

不同帧的 `provenance_id` 不同；同一测量跨 `run_id` 重复导出时，该标识保持不变，便于本体去重。`run_id` 只改变观测记录 ID。帧序号仅参与身份计算，不作为质量特征；当前导出器未测量音质，因此 `quality` 为 `{}`，后续可以由音质计算模块添加契约允许的实际测量值。

默认要求 manifest 中每条录音都有分数。只有通用导入器的 `--allow-subset` 能显式允许缺失录音；这是有意的缺失数据设置，不是跳过错误。无论是否允许子集，未知录音 ID、非有限分数、错误形状、重复帧序号、越界时间、冲突的 registry 条目都报错。

工具先完成所有观测的暂存，再替换目标文件；中途输入或推理失败不会发布半个观测文件。两个目标文件各自使用原子替换，不构成跨文件系统事务，因此磁盘故障时仍应检查两者是否配套。已有 registry 可追加不同专家；同一 `detector_id` 的配置必须完全一致，否则需明确使用新的 ID。输入文件不能被输出路径覆盖。

## 2. CFPRF：直接导出原生帧证据

### 作者代码与外部依赖

官方仓库：[ItzJuny/CFPRF](https://github.com/ItzJuny/CFPRF)。接口核对依据：

- [FDN 网络与 XLS-R 加载路径](https://github.com/ItzJuny/CFPRF/blob/main/models/FDN.py)
- [作者评测入口](https://github.com/ItzJuny/CFPRF/blob/main/evaluate_CFPRF.py)
- [数据读取与类别定义](https://github.com/ItzJuny/CFPRF/blob/main/libs/dataloader/data_io.py)
- [目标对齐辅助函数](https://github.com/ItzJuny/CFPRF/blob/main/libs/tool.py)
- [第一阶段训练入口](https://github.com/ItzJuny/CFPRF/blob/main/train_stage1.py)
- [P2SGrad 损失](https://github.com/ItzJuny/CFPRF/blob/main/libs/loss/p2sgrad.py)

上游仓库标注 MIT；模型、SSL 权重和数据集仍应按各自来源的许可使用。本项目没有复制上游网络实现。

请在独立环境中按照作者 README 准备依赖。核对时作者推荐的组合为 Python 3.8、NumPy 1.23.5、PyTorch 1.8.1 + CUDA 11.1、torchaudio 0.8.1，并使用其指定的 fairseq 版本；还需要读取音频的 `soundfile`。旧版 fairseq / torch 与本框架的现代 Python 环境不一定兼容，因此先在外部环境导出 JSONL，再由主框架读取该文件。这里不承诺未经验证的现代 PyTorch 替换方案。

按作者 README 的链接自行准备 FDN 权重，例如 `1FDN_PS.pth`，以及 XLS-R 300M 权重。目录需要满足：

```text
/path/to/CFPRF/models/FDN.py
/path/to/CFPRF/pretrain_models/xlsr2_300m.pt
/path/to/weights/1FDN_PS.pth
```

其中 `pretrain_models/xlsr2_300m.pt` 是上游构造器硬编码的相对路径。即使 FDN checkpoint 含有编码器参数，构造模型时仍会先访问该文件。适配器暂时切换到外部仓库目录完成构造，然后恢复工作目录。

上游位置编码在构造时直接分配到 CUDA，且未作为普通 buffer 注册；因此此适配器仅支持 `--device cuda` / `cuda:N`。在模型构造前会设置当前 CUDA 设备。CPU 输入被明确拒绝。真实运行还需有与该旧版环境匹配的 GPU / 驱动。

### 实际接口与分数含义

适配器调用：

```python
model = CFPRF_FDN(seq_len=1070, gmlp_layers=1)
seg_scores, bd_scores, emb_T, F_BA = model(waveform)
```

`waveform` 是 `[1, N]`、16 kHz、单声道 float32 波形。适配器使用 `seg_scores`，预期形状为 `[1, T, 2]`，严格加载 state_dict，并验证实际输出帧数。上游类别 0 是伪造，类别 1 是真实，因此导出：

```text
score[t] = seg_scores[0, t, 0] - seg_scores[0, t, 1]
```

FDN 的分类头是线性层，但第一阶段使用基于 one-hot 均方误差的 P2SGrad 损失。因此这里记录为 **未经校准的 raw margin**，不将它称为严格的 log-odds，也不将它误作概率。`bd_scores` 是边界任务输出，本适配器不把它当成第二个伪造专家。

### 原生时间轴与输入限制

本接口按已核对的 XLS-R 卷积配置使用 320 个样本的步长和 400 个样本的初始感受野：

```text
T = floor((N - 400) / 320) + 1
frame[t] = [0.02 * t, 0.02 * (t + 1)) seconds
```

这是帧标签的时间格约定，不表示预测只看这 20 ms。编码器和后端使用上下文，邻帧也并非独立证据。1 秒 / 16000 样本音频应得到 49 帧，最后一格结束于 0.98 秒，剩余尾部保持未观测。适配器不依赖真值数组长度、不将 49 帧拉伸成 50 帧、不把未覆盖尾部补成真实或伪造。

输入必须满足：

- 文件与 manifest 都为 16 kHz，且文件为单声道；有需要时应在导出前显式预处理并更新 manifest。
- 长度为 400 至 340000 个样本，即 25 ms 至 21.25 秒。
- manifest 时长与文件长度相符，允许不超过一个样本的舍入误差。
- 波形必须有限；不进行隐式归一化、重复填充、裁剪或重采样。

上游评测脚本有跳过超长录音和按目标长度截齐分数的步骤；本适配器不调用这些评测函数。超过上述界限会直接报错并停止导出。没有实现分块拼接：分块会改变上下文，需要另行定义重叠区与边界处理并验证后再启用。原生输出形状与公式不同也会报错，避免在仓库 / checkpoint 不兼容时猜测对齐。

### 命令

在已配置好的外部 CFPRF 环境中，从本项目根目录执行：

```bash
python adapters/export_cfprf.py \
  --repo /path/to/CFPRF \
  --checkpoint /path/to/weights/1FDN_PS.pth \
  --manifest /path/to/manifest.jsonl \
  --out /path/to/cfprf_observations.jsonl \
  --registry /path/to/detectors.json \
  --device cuda:0 \
  --detector-id cfprf_fdn \
  --family xlsr300m \
  --run-id cfprf-fdn-v1
```

使用上游预训练 checkpoint 做基准时，还必须核查它的训练数据与当前评测切分是否重叠。适配器能记录文件来源，无法仅由权重文件推断它见过哪些数据。

## 3. 通用 NumPy / CSV 帧分数导入

此工具只依赖 Python 标准库和 NumPy，不依赖 torch、fairseq 或 GPU。它接收真实专家已经导出的帧预测，不能将整句预测自动变成局部预测。

### 支持的文件

| 格式 | 分数结构 | 时间信息 |
|---|---|---|
| `.npy` | 一条录音的数值数组 `[T]` 或 `[T, 2]`，必须给 `--recording-id` | 必须明确 `--frame-hop-sec`，可给 `--frame-start-sec` |
| `.npz` | 每个键是 `recording_id`，值为数值数组 `[T]` 或 `[T, 2]` | 同上；整个导入使用一个固定步长 |
| `.csv` + grid | 每行一帧，列为 `recording_id,frame_index,score`，或两类 `score0,score1` | 每条录音的 `frame_index` 必须从 0 连续递增 |
| `.csv` + intervals | 每行一帧，列为 `recording_id,start_sec,end_sec,score`，或两类分数 | 使用明确的区间，要求有序、非重叠且不越界 |

CSV 可以有额外列，但工具不将它们用于推理。不能指定监督列名作为分数列。NumPy 使用 `allow_pickle=False`，不接受对象数组和 `.pkl`。不同步长或不同起点的结果可用明确区间 CSV 表达，或分别导入为真实定义的专家输出。

两类原始 margin 示例：

```bash
python adapters/import_frame_scores.py \
  --manifest /path/to/manifest.jsonl \
  --scores /path/to/native_scores.npz \
  --checkpoint /path/to/expert_checkpoint.pt \
  --out /path/to/expert_observations.jsonl \
  --registry /path/to/detectors.json \
  --detector-id partialspoof_multi_r2 \
  --family wav2vec2_large_partialspoof \
  --run-id partialspoof-native-v1 \
  --input-score-type margin \
  --layout two-class \
  --fake-class-index 0 \
  --frame-hop-sec 0.02
```

若分数只有一列、原本越大越真实，使用 `--layout scalar --direction higher-is-bonafide`。只有确实为概率时才选 `--input-score-type probability`。不便在导入机器上存放大权重时，可用已经核验的 `--checkpoint-sha256 <64位摘要>` 代替 `--checkpoint`；这只记录身份，不校验该机器上不存在的权重文件。

显式时间 CSV 改用 `--time-mode intervals`，并省略 `--frame-hop-sec` 和 `--frame-start-sec`。工具不根据标签、音频总时长或帧数倒推缺失的 hop。

## 4. PartialSpoof 官方输出桥接

官方仓库：[nii-yamagishilab/PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof)。主要依据：

- [多分辨率模型 README](https://github.com/nii-yamagishilab/PartialSpoof/blob/main/03multireso/README.md)
- [模型 forward、分类输出与对齐代码](https://github.com/nii-yamagishilab/PartialSpoof/blob/main/03multireso/model.py)
- [预训练模型下载脚本](https://github.com/nii-yamagishilab/PartialSpoof/blob/main/03multireso/01_download_pretrained_models.sh)
- [官方分数存储工具](https://github.com/nii-yamagishilab/PartialSpoof/blob/main/project-NN-Pytorch-scripts.202102/core_scripts/other_tools/debug.py)

外部环境依赖作者固定版本的 s3prl 与 project-NN-Pytorch-scripts，且模型中有需按本地目录调整的 `PS_PATH`。README 中有预训练模型与 SSL 文件位置说明。主仓库标注 BSD-3-Clause，子组件与权重仍应分别核查。

已核对的多分辨率设置中，分辨率键 `2, 4, 8, 16, 32, 64` 对应 `20, 40, 80, 160, 320, 640 ms`。它们是同一结构的相关输出，不应假定为六个独立检测器。其 P2S 分类分数使用类别 0 为伪造、类别 1 为真实，接入两列时应使用 `margin` 和 `fake_class_index=0`。

**当前提供的是标准分数桥接接口，不是已完成的 PartialSpoof 无标签推理封装。** 官方 `model.forward` 的评测分支会读取目标标签，并在 `_prepare_segcon_target_ali` 中对齐 / 截断；它把结果保存到 `*_score_ali_2_*.pkl` 等文件，而不是直接返回一个纯预测张量。无标签部署需要在上游模型中从目标对齐之前导出原生帧分数，并保留该版本的时间格。不能因为之后的 CSV 导入器不读标签，就声称上游预测链路没有用过标签。

已有官方基准评测输出可以做可追溯的格式桥接，但必须注明它是作者的已对齐评测产物。已核对的 `score_ali` pickle 保存为多个数组块的列表，每行依次为 `recording_id, score0, score1, target`。以下是将**自己生成、来源可信的**此类文件转成纯预测列 CSV 的独立小例子；不复制模型实现，也不使用最后一列标签：

```python
import csv
import pickle
from collections import defaultdict
import numpy as np

# pickle 仅用于自己已经信任的上游评测输出；通用导入器不加载它。
with open("own_score_ali_2.pkl", "rb") as stream:
    blocks = pickle.load(stream)
if not isinstance(blocks, list):
    raise ValueError("Expected the upstream list-of-array-blocks format")
counts = defaultdict(int)
with open("partialspoof_r2.csv", "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["recording_id", "frame_index", "score0", "score1"])
    for block in blocks:
        rows = np.asarray(block)
        if rows.ndim < 2 or rows.shape[-1] != 4:
            raise ValueError("Expected four columns in every upstream block")
        for row in rows.reshape(-1, 4):
            rid = str(row[0])
            writer.writerow([rid, counts[rid], float(row[1]), float(row[2])])
            counts[rid] += 1
```

然后用第 3 节命令，将 `--scores` 改为该 CSV，并针对 `r2` 指定 `--frame-hop-sec 0.02`。这个转换不会恢复已被上游截掉的帧，不能修复或消除上游的目标依赖。正式无标签实验应优先导出目标对齐之前的原生 NumPy 数组。若使用其他版本的模型或 dump 格式，应先核对实际代码和少量输出，不能机械套用列顺序。

## 5. AASIST：整句 / 窗口证据的边界

官方仓库：[clovaai/aasist](https://github.com/clovaai/aasist)，标注 MIT。依据为 [模型](https://github.com/clovaai/aasist/blob/main/models/AASIST.py)、[推理入口](https://github.com/clovaai/aasist/blob/main/main.py)、[数据处理](https://github.com/clovaai/aasist/blob/main/data_utils.py) 和 [配置](https://github.com/clovaai/aasist/blob/main/config/AASIST.conf)。

AASIST 的模型前向返回 `(last_hidden, output)`，分类输出为 `[B, 2]`，不是 `[B, T, 2]`。作者设置类别 0 为伪造、1 为真实；官方评测使用第二列作为真实方向分数。数据处理将长音频裁为前 64600 个样本，短音频重复填充到该长度，在 16 kHz 时约为 4.0375 秒。

因此，AASIST 是整句或输入窗口级证据。不能把一个分数重复到每个 20 ms 帧，声称获得了局部检测器。当前通用导入器的 registry 固定为 `scope=local`，不提供 AASIST 全局观测适配；以后接入时应另写有明确 `scope` 和实际输入支持范围的适配器。若做滑窗，则观测区间应与输入窗口一致，并单独验证窗口大小、填充、重叠与校准带来的影响。

## 6. 运行验证与后续接入门槛

在本项目根目录可运行不依赖外部权重的验证：

```bash
python -B -m unittest discover -s tests -p test_adapters.py -v
python -B adapters/import_frame_scores.py --help
python -B adapters/export_cfprf.py --help
```

接口测试覆盖两类 margin 方向、概率约束、原生时间格与未覆盖尾部、CSV / NumPy 转换、manifest 监督字段改变后输出保持一致、逐帧稳定来源标识、缺失记录、重复索引、错误 registry、输入覆盖保护，以及 CFPRF 的形状 / 采样率 / 长度 / CUDA / 缺文件检查。端到端契约测试将两条录音的多帧输出实际交给 `ontology.validate_observations` 和 `deduplicate_observations`：不同帧全部保留，同一测量跨 run 重复导出才合并。

真实专家启用前，应先用少量有许可的音频确认：加载的 checkpoint 与外部代码版本匹配；实际输出形状满足原生时间映射；类别方向与作者评测一致；短音频、长音频和异常输入会按说明报错。随后才扩展到完整切分并计算检测指标。这里的接口测试不能替代这些真实权重验证，也不能据此声称某种本体结构已经改善检测效果。
