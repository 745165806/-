# OntoForensics：本体引导的整段伪造语音检测

判断**整段录音的真假**，用多个语音检测专家提供证据，通过具有明确关系类型、来源记录和校准过程的融合器，研究准确性、跨数据集泛化及可解释性。

当前交付研究代码和实验矩阵。LFCC-CNN 已通过小规模真实音频训练、整段推理和融合评测；AASIST 提供可训练实现；SSL 专家需要兼容的 fairseq 环境和 XLS-R 权重。**完整三专家、多随机种子和跨数据集实验尚未运行，不能据此宣称本体融合提升了检测性能。**

## 安装

Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[utterance,ontology]'
python -m unittest discover -s tests -v
python -m ontoforensics.utterance --help
```

`torch` 应按机器的 CUDA 环境安装。本项目不自动下载语料和权重。只加载可信来源的 PyTorch checkpoint 和融合 `joblib` 文件。SSL 的旧版 fairseq 兼容问题与分环境操作见 [整段检测指南](docs/UTTERANCE.md)。

## 准备数据

读取已有的统一 CSV 索引。ASVspoof2019 LA 官方 train 用于训练专家，官方 dev 按说话人划分五个互斥角色：专家选择、融合训练、融合选择、概率校准、阈值选择。官方 eval 和外部语料只作为测试集。

```bash
export CORPUS_INDEX=/path/to/fakedata/total_label
python -m ontoforensics.utterance prepare \
  --indexes "$CORPUS_INDEX/ASVspoof2019_LA_train.csv" \
            "$CORPUS_INDEX/ASVspoof2019_LA_dev.csv" \
            "$CORPUS_INDEX/ASVspoof2019_LA_test.csv" \
            "$CORPUS_INDEX/InTheWild_unspecified.csv" \
  --output manifests/utterance.jsonl
```

也可输入索引目录或 `index.csv`，表示读取目录清单内所有语料。建议先显式选择少数语料。排除记录写入 `.excluded.jsonl`，划分和检查范围写入 `.report.json`。`--check-audio` 会解码并计算内容哈希，不加时只检查文件和元数据。未知说话人的外部数据按录音分组，不声称已验证跨语料说话人隔离。

## 单专家闭环

```bash
python -m ontoforensics.utterance train-expert \
  --manifest manifests/utterance.jsonl --config configs/utterance/lfcc_cnn.json \
  --output runs/lfcc_cnn --device cuda:0

python -m ontoforensics.utterance export \
  --manifest manifests/utterance.jsonl --checkpoint runs/lfcc_cnn/best.pt \
  --output runs/lfcc_cnn/scores.jsonl --device cuda:0

python -m ontoforensics.utterance fit-fusion \
  --manifest manifests/utterance.jsonl --scores runs/lfcc_cnn/scores.jsonl \
  --ontology configs/utterance/lfcc_only.json --mode mean \
  --output runs/lfcc_cnn/fusion.joblib

python -m ontoforensics.utterance evaluate \
  --manifest manifests/utterance.jsonl --scores runs/lfcc_cnn/scores.jsonl \
  --model runs/lfcc_cnn/fusion.joblib --output runs/lfcc_cnn/test
```

`select --manifest ... --per-class 2 --output manifests/pilot.jsonl` 可在固定角色内取小样本；搭配 `configs/utterance/lfcc_smoke.json` 与 `--device cpu` 验证流程。小样本结果不能作为论文性能比较。

## 多专家实验与解释

专家为 AASIST、SSL anti-spoofing 和 LFCC-CNN。`configs/utterance/ontology.json` 声明专家与机制族。这是可审计的分组假设，不代表专家之间统计独立。

| 模式 | 作用 |
|---|---|
| `mean` | 标准化专家 margin 的平均，再独立校准 |
| `flat_lr` | 分数、有效性 mask、实测质量的线性基线 |
| `mlp` | 使用相同基础输入的非线性基线 |
| `pairwise_lr` | 每个专家对分别学习交互系数 |
| `relation_lr` | 按同族/异族关系池化并共享系数，加入质量交互 |

```bash
export ONTO_XLSR_CHECKPOINT=/path/to/xlsr2_300m.pt
python -m ontoforensics.utterance.experiment \
  --manifest manifests/utterance.jsonl --configs configs/utterance \
  --output runs/experiments --device cuda:0 --seeds 17 42 2026 --bootstrap 1000
```

矩阵包含五种融合、关系/质量消融、逐一移除专家和单专家对照，另有按已知说话人/录音分组的配对 EER bootstrap。按数据集分别报告 EER、AUROC、AP、F1、FPR/FNR、Brier、NLL 和覆盖率。模型报告保留 MLP 收敛警告。

预测文件含专家权重/配置哈希、缺失专家、机制族及线性模型的原始和校准后 logit 贡献。解释说明当前模型如何组合观测，不是因果证明。缺少任一必需专家时拒判。无标签新录音使用 `role="unspecified"`、`label=null`，运行相同专家导出后使用 `predict`，参数与 `evaluate` 相同。

```bash
python -m ontoforensics.utterance export-graph \
  --predictions runs/lfcc_cnn/test/predictions.jsonl --output runs/lfcc_cnn/evidence.ttl
```

证据图不导出真值。融合器读取显式数值关系配置，RDF 用于审计；没有实现 OWL 自动推理、因果推理、CRF 或 RDF 图神经网络。

## 文档

- [整段检测指南](docs/UTTERANCE.md)：数据契约、SSL 环境、恢复训练与输出。
- [实验方案](docs/UTTERANCE_EXPERIMENTS.md)：主要指标、对照与失败条件。
- [验证记录](VALIDATION.md)：实际检查及未完成实验。
- [第三方声明](THIRD_PARTY.md)：随附专家源码及原许可证。
- [旧版局部检测](docs/LOCALIZATION_LEGACY.md)：保留的原命令行和测试。

完整训练的资源消耗及泛化结论须通过真实实验验证。语料、权重、私有路径和运行缓存不随源码发布。
