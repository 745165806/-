# 原局部伪造框架（0.1.0 存档）

本页保留旧版局部检测说明；有关未实现 AASIST 等描述仅指原局部接口。整段检测为当前主要任务，见 [新版入口](../README.md) 和 [整段指南](UTTERANCE.md)。

这是一个可运行、可审计的研究起点：使用已有语音专家的缓存分数，把观测的时间、来源、关系类型和缺失状态组织起来，训练可校准的融合模型，输出整句判断、候选编辑区间及对应证据。

**当前实现是“类型化证据 + 按关系类型共享参数的特征融合”，没有实现 CRF、GNN 或自动因果推理。代码和演示跑通不等于在真实语音上验证了效果。** `runs/demo` 使用人为模拟的数值观测，不生成音频，也不运行真实语音模型。真实 checkpoint 推理尚未在本交付环境运行。

## 1. 先用五分钟理解交付物

| 内容 | 当前实现 |
|---|---|
| 数据 | PartialEdit v1.1 CSV / speaker split 导入，配对 VCTK 原录音；原始录音与说话人分组隔离 |
| 音频 | 显式转成 16 kHz 单声道 PCM16；不裁静音、不做响度归一化；测量局部 RMS / 削波比例 |
| 真实专家 | 外部 CFPRF FDN 原生帧分数导出代码；独立旧版 CUDA 环境运行，真实权重尚未实测 |
| 其他专家 | 数值 NPY / NPZ / CSV 标准接口；PartialSpoof 原生推理仍需接钩子，不伪称已经封装 |
| 本体 | RDF/OWL 小型领域词表，对齐 PROV-O / SOSA；SHACL 与 Python 结构检查 |
| 来源处理 | 验证副本与同一输出来源后去重；不把相关模型当成已知独立证据 |
| 融合 | `mean`、`flat_lr`、`flat_mlp`、`relation_lr` 四组基线 |
| 关系 | 同族 / 异族成对关系按类型池化并共享系数；局部绝对偏离；分数与实测质量交互 |
| 校准 | train 拟合检测器尺度、特征尺度与融合；calibration 校准；dev 定阈值；test 冻结评测 |
| 输出 | 录音概率、局部候选区间、覆盖率、拒判、证据 ID、线性特征贡献 |
| 评测 | 帧级秒加权指标、整句 EER/AUROC/AP/FPR/召回/Brier/NLL、按条件分组、源录音 bootstrap |

代码只使用白名单中的实际观测数值。`label`、编辑区间、真实编辑器、真实 codec 条件、speaker、dataset、文件名和来源 ID 不进入预测特征。标签只用于损失与评测；来源 ID 只用于划分、去重和审计。

## 2. 安装与一条命令运行

在解压后的项目根目录，使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ontology,audio]"
python -m ontoforensics demo --output runs/demo --with-rdf
python -m unittest discover -s tests -v
```

Windows 激活命令为 `.venv\Scripts\activate`。没有网络下载依赖时，请先在同版本环境准备 wheel；代码本身不自动下载数据和权重。

`demo` 生成 208 条模拟录音记录及三个模拟专家的观测，完成四组融合的训练、校准、阈值选择、测试输出与一个小型 RDF/SHACL 示例。每个原始源的真实 / 编辑和两个模拟条件变体始终处于同一 split。所有文件都带 `simulation_only` 或 `is_simulation` 标记。演示中的音频路径是明确不存在的占位路径，不能用于真实推理或 `--check-audio`。

主要输出：

- `runs/demo/fixture/`：模拟 manifest、观测和 registry。
- `runs/demo/<mode>/model.json`：纯 JSON 数值模型、尺度、校准器、冻结阈值、已见分组。
- `runs/demo/<mode>/train_report.json`：划分检查、特征名、开发指标、优化警告。
- `runs/demo/<mode>/test/`：预测、帧分数、指标及校验报告。
- `runs/demo/evidence_sample.ttl`：可读的证据实例图。
- `runs/demo/demo_summary.json`：工程演示汇总，不能作为论文结果表。

若 MLP 达到迭代上限，警告会写入训练报告。正式对比必须检查收敛，并在开发集上给各基线合理的调参预算，不能把未收敛的小网络当成强基线。

## 3. 第一轮真实数据：先完成单专家闭环

### 3.1 准备数据并固定划分

下载 [PartialEdit v1.1](https://zenodo.org/records/18829689) 的 CSV、speaker ZIP 和 E1/E2 音频，并按发布说明准备配对的 VCTK 原音频。首轮先使用 E1/E2；Codec 扩展留给有真实转码负样本的受控实验。详细格式与已核验的命名差异见 [DATA_IMPORT.md](DATA_IMPORT.md)。

```bash
python -m ontoforensics import-partialedit \
  --root /data/PartialEdit \
  --csv /data/PartialEdit/PartialEdit_E1E2.csv \
  --speaker-splits /data/PartialEdit/PartialEdit_spk.zip \
  --bona-fide-root /data/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --mic mic1 --subsets E1 E2 \
  --output manifests/partialedit.jsonl

python -m ontoforensics split-dev \
  --manifest manifests/partialedit.jsonl --fraction 0.5 --seed 17 \
  --output manifests/partialedit_split.jsonl

python -m ontoforensics select-pilot \
  --manifest manifests/partialedit_split.jsonl \
  --groups-per-split 100 --require-paired \
  --min-duration-sec 0.025 --max-duration-sec 21.25 --seed 17 \
  --output manifests/pilot.jsonl

python -m ontoforensics prepare-audio \
  --manifest manifests/pilot.jsonl --audio-dir data/pilot_16k \
  --output manifests/pilot_16k.jsonl
```

`select-pilot` 在既定 split 内抽取原始录音组，保留组内所有变体；长度不合格时整组排除并记入报告。`--require-paired` 只保留同时有真实与编辑样本的组。它产生的是试验子集，不是官方全量基准。长度限制来自当前 CFPRF 适配器的支持范围，不能把删掉的长音频当作成功检测。

导入默认严格失败。若原语料缺少特定麦克风或源录音，应先确认文件和版本；确需排除时显式用 `--non-strict`，检查同名 `.import_report.json` 的完整失败清单，再使用配对筛选。不可静默换麦克风、猜时间偏移或随机跨原录音划分。

`prepare-audio` 对两类使用相同处理，保持时间原点，重采样长度变化不超过一个输出样本。原始文件保留。输出目录已有同名音频时会报错，避免混用过期预处理缓存；重跑时使用新目录。

### 3.2 在外部专家环境导出 CFPRF

本框架的融合环境与旧版 CFPRF / fairseq 环境分开。请先读 [EXPERTS.md](EXPERTS.md)，从作者仓库准备网络、FDN checkpoint 和硬编码路径所需的 XLS-R checkpoint。下面命令使用的是外部环境的 Python，并从本项目根目录执行：

```bash
python adapters/export_cfprf.py \
  --repo /models/CFPRF \
  --checkpoint /models/CFPRF/checkpoints/1FDN_PS.pth \
  --manifest /absolute/path/to/manifests/pilot_16k.jsonl \
  --out /absolute/path/to/cache/cfprf.jsonl \
  --registry /absolute/path/to/cache/detectors.json \
  --device cuda:0
```

这一适配器实际调用外部网络，导出未经校准的 `class0 − class1` margin，按已核对的原生 20 ms 时间格记录。它不读取真值来对齐预测，不补齐未观测尾部，不静默跳过错误。由于当前没有执行真实模型推理，第一次应先用少量符合长度要求的音频验证实际输出形状及类别方向，再扩大到试验子集。

### 3.3 回到融合环境，测量质量并训练

```bash
python -m ontoforensics annotate-quality \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/cfprf.jsonl --detectors cache/detectors.json \
  --output cache/cfprf_quality.jsonl

python -m ontoforensics validate \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/cfprf_quality.jsonl --detectors cache/detectors.json \
  --check-audio

python -m ontoforensics train \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/cfprf_quality.jsonl --detectors cache/detectors.json \
  --mode flat_lr --step-sec 0.04 --target-fpr 0.05 \
  --output runs/pilot_single

python -m ontoforensics evaluate \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/cfprf_quality.jsonl \
  --model runs/pilot_single/model.json --split test --bootstrap 200 \
  --output runs/pilot_single/test
```

单专家阶段检验数据、标签、时间格、校准和指标，不检验多模型创新。RMS 与削波率只是可复现的低成本协变量，不是 codec 分类器；代码不会凭空生成 SNR。

## 4. 第二轮才检验关系融合

通过 `adapters/import_frame_scores.py` 接入真正的第二个专家，用不同 `detector_id`、正确共享来源族和原生时间格导出。两个专家可以先验证普通融合；要检验本实现“按同族/异族关系共享参数”的增量，建议至少三个有真实输出的专家，并同时存在同族与异族关系。只有一个专家对时，关系类型不能提供充分的参数共享对照。PartialSpoof 多分辨率输出同源，不能按六个独立投票计数。无标签导出需要在作者依赖目标对齐的步骤之前取原生输出；本项目没有把这一步假称为已完成。

两个导出接口可以向同一个 registry 添加不同检测器。合并后验证，再运行相同数据的四组对照：

```bash
python -m ontoforensics merge-observations \
  --manifest manifests/pilot_16k.jsonl \
  --inputs cache/cfprf.jsonl cache/second_expert.jsonl \
  --detectors cache/detectors.json --output cache/merged.jsonl

python -m ontoforensics annotate-quality \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/merged.jsonl --detectors cache/detectors.json \
  --output cache/merged_quality.jsonl
```

分别将上一节训练命令的 `--mode` 设为 `mean`、`flat_lr`、`flat_mlp`、`relation_lr`，输出到不同目录。各模型使用相同的分数、质量、缺失 mask、时间上下文和划分；只有关系模型增加显式类型池化、绝对偏离与交互特征。

三个消融配置已经提供：

- `--relations configs/no_same_family.json`：删除同族关系项。
- `--relations configs/no_temporal.json`：删除绝对局部偏离，仍保留基线共有的上下文均值/标准差。
- `--relations configs/no_quality_interaction.json`：删除质量交互，仍保留质量主效应。

先选定主要指标与失败条件，完整计划见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。关系版若只超过等权平均、没有超过充分训练的 MLP，不能据此断言本体提供了独特的预测优势。

## 5. 本体图与预测解释

```bash
python -m ontoforensics export-graph \
  --manifest manifests/pilot_16k.jsonl \
  --observations cache/cfprf_quality.jsonl --detectors cache/detectors.json \
  --output runs/pilot_single/evidence.ttl
python -m ontoforensics validate-rdf --graph runs/pilot_single/evidence.ttl
```

RDF 是同一证据表的可审计视图；数值模型直接读取经过校验的 JSONL / registry / `relations.json`。修改 Turtle 标签不会自动重写分类器。SHACL 检查结构条件，不把检测分数变成真值。本体文件、本体与实现边界、来源复制关系等见 [ONTOLOGY.md](ONTOLOGY.md)。

对无标签新数据，manifest 仍需可审计的录音/源/说话人字段（未知身份可用明确的逐录音占位 ID，不能伪称已经验证跨说话人隔离），设置 `split="unspecified"`、`label=null`、编辑区间为空，然后用 `predict`。任何标签和真实攻击条件都不必提供。

```bash
python -m ontoforensics predict \
  --manifest manifests/new_unlabelled.jsonl --observations cache/new_observations.jsonl \
  --model runs/pilot_single/model.json --output runs/new_prediction
```

`candidate_intervals` 是固定网格上超过阈值的连续区间合并，不是完整 PRN/CRF 定位结果。只有被至少一个局部专家的有效时间区间完整覆盖的网格才可输出；部分覆盖的末格保持未观测，重叠窗口按时间并集检查，避免拉长预测。报告的覆盖率因此是保守的输出网格覆盖率。无局部证据时返回 `review` 与空概率；有证据但低置信时也可拒判。该阈值是演示性选择性预测规则，不提供未知分布风险保证。特征贡献只解释当前线性模型的分数，不是伪造因果证据，也不恢复原始文字。

## 6. 规模与当前边界

融合端默认把记录和帧特征放入内存。建议先按每个 split 50–100 个源录音组做闭环，再评估时间、内存与专家吞吐；全量数据需改成分片缓存/批处理，不能直接按小型演示推断全量资源。GPU需求取决于外部专家；本交付没有测量真实推理耗时或显存，不给未经测试的吞吐承诺。

当前未完成：完整 PartialSpoof 数据导入与无标签模型封装、AASIST 全局接口、神经 codec 的配对增广、长录音分块、CRF/GNN、官方 SegmentEER/RangeEER/区间 mAP、外部语料真实实验。这些都是后续任务，不隐藏在“已经支持”的表述中。

代码许可证见 [LICENSE](../LICENSE)；第三方仓库、权重、数据许可各自独立，见 [THIRD_PARTY.md](../THIRD_PARTY.md)。验证记录见 [VALIDATION.md](../VALIDATION.md)。
