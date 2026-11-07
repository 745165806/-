# 整段语音检测操作指南

## 输入契约

`prepare` 读取统一 CSV：`dataset,split,id,native_id,path,storage,row_index,label,source_fields_json`。`label=0` 表示真实、`label=1` 表示伪造；未知标签不猜测，写入排除报告。`path` 是可访问的文件路径。`storage` 为 `file`、`parquet_embedded` 或 `arrow_embedded`；嵌入式音频的 `row_index` 为从零开始的全局行号，`audio` 列是含 `bytes` 的结构体。Parquet 只解码对应行组；Arrow 逐批读取，不一次展开整片音频。

本机数据可通过 `CORPUS_INDEX=/path/to/fakedata/total_label` 指定；目录和数据不会随代码发布。`prepare` 也支持该目录下的 `index.csv` 清单。每条记录保留索引 SHA256，不将文件名、说话人、语料、标签或攻击 ID 输入预测特征。

音频转为单声道 16 kHz，不裁静音、不作响度归一化。训练随机裁取一个窗口；推理覆盖全部录音，以实际样本长度加权平均窗口 logits，最后不足一窗时循环补齐。补齐仍可能影响短录音分数，应检查时长分组表现。实测质量为 RMS dB 和削波比例，不估造 SNR 或真实 codec 标签。

## 划分与泄漏检查

默认只有 ASVspoof2019 LA 的官方 train 进入 `expert_train`。dev 的说话人按约 20/35/15/15/15 分到 `expert_select/fusion_train/fusion_select/calibration/threshold`，每组必须含两类。eval 与外部语料保持为 `test`。

`validate` 检查 ID、音频位置和组的跨角色重复。`--check-audio` 另计算重采样后波形的内容哈希；它能识别完全相同的波形，不能识别经过压缩或增广的近重复。未知说话人的语料以原始录音 ID 代替组 ID，因此不能据此保证跨语料说话人隔离。

融合工件保存拟合过的 ID、组、音频位置与可用内容哈希。独立评测会检查这些集合，不能通过给训练音频换 ID 绕过检查。融合拟合完全排除 `test` 观测。每个专家必须提供固定的权重/配置哈希、整段作用域和明确的分数方向；混用版本、相冲突的重复观测、将同一模型登记为多个专家都会失败。

## 专家和环境

LFCC-CNN 和 AASIST 可在 Python 3.10 + PyTorch 2.6 环境进行 CPU 验证。正式训练推荐可用的 CUDA 环境。AASIST 和 SSL 的分类头在本训练流程中使用统一的 0=真实、1=伪造标签，导出 `logit[1]-logit[0]`。不要直接把第三方不同标签语义的训练后权重装入本接口。

SSL 使用随附的上游 SSL anti-spoofing 网络及其 fairseq XLS-R 加载器。上游 fairseq commit 为 `a54021305d6b3c4c5959ac9395135f63202db8f1`。它不是现代 PyTorch/Python 的通用即装即用依赖，因此没有放入默认依赖。应在独立环境按上游项目准备 fairseq，并在支持本项目 Python 3.10+ 的环境中验证兼容性；此交付未完成真实 SSL 权重训练验证。设置：

```bash
export ONTO_XLSR_CHECKPOINT=/path/to/xlsr2_300m.pt
python -m ontoforensics.utterance train-expert \
  --manifest manifests/utterance.jsonl --config configs/utterance/ssl.json \
  --output runs/ssl --device cuda:0
```

如果三个专家需要不同环境，分别运行 `train-expert` 和 `export`，再把 JSONL 分数交给融合环境：

```bash
python -m ontoforensics.utterance fit-fusion \
  --manifest manifests/utterance.jsonl \
  --scores runs/aasist/scores.jsonl runs/ssl/scores.jsonl runs/lfcc_cnn/scores.jsonl \
  --ontology configs/utterance/ontology.json --mode relation_lr \
  --output runs/fusion/model.joblib
```

`evaluate` 使用相同 manifest、scores 和 model。单进程 `experiment` 命令需要同一环境能运行所有已配置专家；不满足时使用分环境命令。

## 恢复与缓存

```bash
python -m ontoforensics.utterance train-expert \
  --manifest manifests/utterance.jsonl --config configs/utterance/lfcc_cnn.json \
  --output runs/lfcc_cnn --resume runs/lfcc_cnn/last.pt --device cuda:0
```

保留同一运行目录的 `best.pt`。恢复检查配置、训练/选择数据指纹，恢复优化器、随机状态和历史；不允许换数据后继续旧实验。权重通过临时文件替换，历史记录为 `history.jsonl`。批大小以 `effective_batch` 为优化器步长，`micro_batch` 控制单次显存占用。初次运行可调整配置，但恢复时不得变动配置。

矩阵运行的 `completed.json` 校验输入 manifest、代码、配置和分数哈希，不接受过期缓存。实验产生 `results.jsonl` 和逐种子的 `paired_bootstrap.json`。没有可用 CUDA 时默认矩阵会明确失败；小规模 CPU 验证须显式选择 `--device cpu`。

## 输出和解释

`model.joblib.report.json` 包含候选正则化、验证 NLL、阈值、来源与收敛警告。阈值只由 `threshold` 角色选择，默认目标开发 FPR 为 5%，不是测试/未知分布的风险保证。

`predictions.jsonl` 中 `contributions` 加 `intercept` 重建未校准 logit；`calibrated_contributions` 加 `calibrated_intercept` 再经 sigmoid 重建最终概率。仅线性融合提供这种加性解释，MLP 不提供伪造的线性贡献。原始分数和其交互是相关特征，贡献不等同于独立证据或物理因果。

缺少任一专家时输出 `status="missing_expert"`、空概率和缺失清单。指标只使用有效且有标签的记录，并另报 `total/usable/coverage`，避免把拒判样本从分母中隐藏。risk-coverage 使用冻结阈值，不能通过测试集重新选阈值。

`export-graph` 导出整段预测的 PROV/SOSA 观测、模型来源及线性贡献。它使用独立的 utterance 词表，不能将旧版局部证据的 SHACL 规则直接当作本图的验证通过记录。

## 规模

发布前可用真实音频运行有界验证：

```bash
python scripts/smoke_utterance.py --manifest manifests/utterance.jsonl \
  --output runs/release_smoke --device cpu
```

它按每个角色、数据集和类别抽取最多两条录音，计算内容哈希，运行 LFCC-CNN 与 AASIST 各一轮训练、全录音推理、11 组融合/消融和少量 bootstrap。输出 `summary.json` 明确标记为工程验证；不会运行 SSL，也不能替代全量实验。

融合和分数导出仍在内存保存记录列表；没有宣称已支持任意规模的流式实验。首先用 `select` 和小样本验证，再测量吞吐与内存。三专家全量训练、XLS-R 兼容性、跨语料内容/说话人去重和多随机种子的统计结论仍需在真实研究环境中完成。
