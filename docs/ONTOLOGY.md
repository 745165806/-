# 证据本体、输入校验与可执行关系

这个实现把本体用于**定义证据类型、检查关系是否合法、约束特征生成**。检测器输出先形成带来源的结果记录，再由训练得到的融合模型判断局部伪造。它是轻量的关系特征与线性融合基线，**没有实现 CRF、GNN、概率 OWL 推理或因果识别**。

## 1. 文件与职责

| 文件 | 职责 |
| --- | --- |
| `ontology/forensics.ttl` | 轻量 OWL/RDFS 词汇及 PROV/SOSA 对齐 |
| `ontology/shapes.ttl` | 可执行 SHACL 约束，校验导出的 RDF |
| `ontology/relations.json` | 与 `features.py` 对齐的关系开关、特征含义及限制 |
| `src/ontoforensics/ontology.py` | JSON 校验、来源副本去重、Turtle 导出、可选 SHACL 校验 |
| `src/ontoforensics/resources/` | 安装包携带的同版本资源副本，供 wheel 安装后使用 |

纯 Python 校验、去重和 Turtle 导出只依赖标准库。额外的 RDF/SHACL 校验需要：

```bash
python -m pip install 'rdflib>=6' 'pyshacl>=0.25'
```

没有可选依赖时，`validate_rdf` 明确返回 `validated: false, conforms: null, status: skipped` 和安装命令，不能将它解释为已经通过校验。资源查找优先使用源码目录的 `ontology/`，然后使用包内 `resources/`，不依赖当前工作目录。

## 2. 类和语义边界

| 类 | 对齐 | 含义 |
| --- | --- | --- |
| `AudioRecording` | `prov:Entity` | 实际接收到的录音 |
| `TimeWindow` | `prov:Entity` | 半开分析时间窗 `[start,end)`，不是标注中的伪造区间 |
| `DetectorModel` | `prov:Entity` | 注册的检测器/模型配置或工件 |
| `DetectorRun` | `prov:Activity` | 某检测器在某录音上的一次调用 |
| `Observation` | `prov:Entity`、`sosa:Result` | 检测输出或输出缺失的结果记录 |
| `LocalObservation` | `Observation` | 目标为局部分析窗 |
| `GlobalObservation` | `Observation` | 目标为整段录音 |
| `EvidenceFamily` | 项目词汇 | 注册的证据机制族，供关系特征分组 |

**这里的 `Observation` 是结果实体。W3C 的 `sosa:Observation` 是进行观测的活动，二者没有被等同。** `DetectorRun` 可能生成多个窗口的结果，因此当前只对齐 `prov:Activity`，不假定它恰好对应一项 SOSA 观测活动。

关系包括窗口所属录音、结果的生成调用、调用使用的检测器、检测器所属族、已知副本来源、同目标同族关系和时间顺序关系。没有根据 `score > 某阈值` 直接写入“该窗口为伪造事实”的公理。

RDFS 的 domain/range 用于类型推导，不代替数据校验。输入约束由 Python 和 SHACL 执行。通过约束检查只说明结构和来源声明自洽，不证明媒体真实，也不证明检测器可靠。

## 3. JSON 输入合同

`validate_observations(observations, records, detectors)` 接受观测对象列表，以及记录/检测器的对象列表或 ID 到对象的映射。检测器还支持 `{"detectors": [...]}` 格式；记录支持 `{"records": [...]}`。

### 录音与检测器

每个录音至少需要 `recording_id`、正的有限 `duration_sec`、正整数 `sample_rate`。预测时 `label` 可以为 `null` 或缺省，`fake_intervals` 可以缺省。只有提供了非空 `label` 时才要求其值为 0 或 1。

每个检测器需要：

```json
{
  "detector_id": "partialspoof",
  "family": "local_synthesis_trace",
  "score_type": "margin",
  "higher_is_fake": true,
  "scope": "local"
}
```

`score_type` 支持 `logit`、`margin`、`probability`。PartialSpoof 的余弦分数差应称为 `margin`；不因具有正负号就称为 logit。`probability` 分数必须位于 `[0,1]`，但符合这个范围不代表已经校准。可选 `checkpoint_sha256` 必须为 64 位十六进制串；可选 `synthetic` 是审计标志。

### 一条观测

```json
{
  "observation_id": "obs-001",
  "recording_id": "recording-001",
  "detector_id": "partialspoof",
  "run_id": "run-001",
  "start_sec": 0.0,
  "end_sec": 0.5,
  "score": -0.73,
  "status": "valid",
  "provenance_id": "specific-detector-output-001",
  "derived_from": null,
  "quality": {
    "rms_db": -24.1,
    "clipping_fraction": 0.0
  }
}
```

局部检测器要求 `0 <= start_sec < end_sec <= duration_sec`。整句检测器仍使用明确数值边界 `start_sec=0`、`end_sec=duration_sec`；导出 RDF 时其目标是 `AudioRecording`，不会伪装成局部检测结果。JSON 不支持用空边界代表整句，以保持与数值特征接口一致。

`status` 支持 `valid`、`missing`、`not_applicable`、`failed`。`valid` 必须有有限数值 `score`；其他状态必须 `score:null` 或不含 score 字段。**合法的零分和缺失分数不同。** NaN、Infinity、布尔值不能作为分数。

`quality` 只接受实际测得的 `rms_db`、`clipping_fraction`、`estimated_snr_db`；后者缺失时不能伪造 SNR。削波比例限制在 `[0,1]`。质量字典可以为空，字段值为 null 时作为未测得处理。

`label`、`fake_intervals`、`ignore_intervals`、`editor`、`condition`、`metadata` 留在 manifest 中用于监督或审计，不能塞进观测或质量字典。`export_graph` 仅导出录音 ID、时长、采样率和检测证据；不导出上述标签、真值 codec/生成器、speaker、split 或数据集名称。根特征构建器使用明确字段名单，监督目标由单独的 `make_targets` 产生。

## 4. 来源去重的具体约定

`deduplicate_observations(observations)` 返回 `(canonical_observations, audit_report)`。它不修改输入。

1. 完全一致的重复 `observation_id` 记录可以合并；同一 ID 的内容冲突会拒绝。
2. 本基线中 `derived_from` **特指同一输出的副本引用**。引用必须能在当前批次找到，不允许自引用或循环。副本与原件必须有相同录音、时间窗、状态、分数和已测质量。
3. `provenance_id` 标识某一条检测输出。如果用整段 WAV 的 ID 或整次运行 ID给所有窗口作 provenance，会产生冲突并被拒绝。
4. 同一检测器、同一录音/时间窗且结果内容完全相同的重复输出，也会合并。两个无已知来源关系的不同检测器即使恰好同分，也不会被合并。
5. 跨检测器 ID 的副本需要用 `derived_from` 指明原件，避免新增别名改变保留的检测器。完整校验还检查它们的 family、scope、score_type、方向和已提供 checkpoint hash 是否一致。

`prov:wasDerivedFrom` 在通用 PROV 中含义更宽。重新运行、校准、变换分数等活动可能产生新结果，不能仅凭这种宽泛关系认定重复。因此当前 JSON 的窄约定必须遵守；将来若支持通用变换谱系，应增加独立字段和活动节点。

完成去重后，同一检测器/录音/原始时间窗只允许一个规范测量。非等价的多个测量会报错，防止特征表悄悄覆盖其中一条。明确需要多次随机推断的系统应另行设计重复试验模型。

审计报告中的副本数、观测 ID、run ID 和来源 ID **不进入预测特征**。RDF 保留副本以便追溯，并以 `isCanonical=false` 和 `knownDuplicateOf` 指向保留的记录。`knownDuplicateOf` 不采用 `owl:sameAs`，因为原件和副本记录仍是不同实体。

已知副本去重不等于已经解决所有统计依赖。两个相近模型没有明确相同来源时，仍可能有高度相关的错误；同族关系只提供一个需要验证的结构假设。

## 5. 关系如何参与实际特征

`relations.json` 由数值特征构建器读取，不只是 RDF 展示配置。先去重，然后将局部观测按与分析帧的重叠时长加权，形成统一时间网格上的每个检测器分数与可用性 mask。概率分数被裁剪端点、变为 log-odds，再根据方向统一；margin/logit 保持各自含义，原始定向分数裁剪至 `[-20,20]`。去重后，`fit_score_normalization` 只用 train 分区的有效观测，以观测时长为权重拟合每个检测器的均值/标准差，并保存到 `model.json`。在构造成对关系和质量交互前，先应用这些固定的检测器尺度；之后的特征标准化器也只在 train 上拟合。这些尺度变换不构成概率校准。

数值数组的空位置可以暂用零，但必须和 `valid=0` 一起表示。原始观测仍保留 null；局部均值/方差只统计可用观测。整句观测可作为每个局部决策的条件特征，但不会使一个没有局部证据的帧变为可判定，也不作为多个独立的检测票。

| 关系开关 | 当前增加的特征 | 必须理解的限制 |
| --- | --- | --- |
| `same_family` | 将全部同族 pair 按类型汇总为三列：有效 pair 的分数乘积均值、绝对分数差均值、有效 pair 比例 | 每种关系类型共享一套三个系数；关闭此开关移除整个同族块。异族 pair 采用相同汇总，在存在候选边时保留其独立类型块 |
| `temporal_context` | `abs(当前分数 - mask-aware 局部移动均值) × 当前valid`，即局部绝对偏离 | flat 基线已有相同上下文均值/标准差；当前项增加非线性的局部偏离幅度。它不进行序列标签的联合解码，也不推断编辑边界的因果来源 |
| `quality_interaction` | 分数 × 有物理尺度的质量值 × 分数/质量可用性 | 质量的主效应和 mask 已在基础特征中；回归系数由训练得到，不把分数乘某个“可靠度”直接称为概率 |

成对关系采用 **type-pooled（按关系类型汇总）** 设计。对所有不重复的检测器对 `i<j`，由注册的 family 是否相同划分为 `same_family` 和 `different_family`。每一类型只产生以下三列：

- `relation/{type}/joint_mean`：对当前帧中两端均有效的 pair，计算分数乘积的均值。
- `relation/{type}/disagreement_mean`：对同一批有效 pair，计算绝对分数差的均值。
- `relation/{type}/available_fraction`：有效 pair 数除以该类型全部可能的 pair 数。

两个均值的分母是**当前有效 pair 数**，可用比例的分母是**该类型可能的 pair 数**。某帧没有有效 pair 时，两个均值暂用零，并以可用比例零表明缺失；整个注册表中没有该类型候选 pair 时，不生成这一类型块。

每一类型的每项统计量只有一个回归系数，同类型的 pair 通过汇总共享这一套三个系数。检测器各自的分数主效应仍可有不同系数。family 的划分决定汇总哪些边、可用比例的分母以及参数共享方式，因此可以通过改变分族检验类型结构的作用。只重命名 family 而不改变同族/异族划分，则不应改变这些特征。分数乘积和分数差属于数值协变量，不是独立概率票，也不能据此认定检测器错误相互独立。

**只接入一个 CFPRF 专家时，没有检测器 pair，也没有上述成对关系块。** 时间和质量特征仍可运行，但这种实验不能证明多专家融合或多专家依赖处理有效。

质量缩放常数为 RMS 40、削波比例 1、估计 SNR 30，缩放后裁剪至 `[-5,5]`，不是从测试标签拟合的常数。每检测器分数尺度及下游估计器的特征标准化都只在训练数据上拟合。

**把三个开关全关不一定等于 flat 模式**：只要存在异族候选 pair，relational 模式仍有异族类型汇总块。完整对照需要使用根模型提供的 flat/relational 模式及消融设置。

RDF 中同族关系记录同一原始目标上的关系；`nextTemporalObservation` 记录同一检测器在同一录音中的原始窗口顺序。数值编译器在对齐的帧网格上执行关系和局部邻域计算，**不在运行时加载整个 RDF 图进行消息传递**。SHACL 检查关系结构，不能替代检测器校准或模型训练。

## 6. SHACL 与 Python 分工

Python 在建图前检查字段、有限数值、范围、注册表引用、来源环、来源矛盾和唯一规范测量。SHACL 再检查：

- 录音时长/采样率、窗口边界及所属录音。
- 结果恰属局部或全局类别，目标与运行所使用录音一致。
- valid 必须有 score，其他状态没有 score；概率分数范围合法。
- 副本不自引用/成环，指向相同目标、状态和分数的规范记录。
- 同族关系的目标、注册族和规范状态一致。
- 时间顺序关系在同一录音/检测器内，目标时间严格前进。

`validate_rdf` 额外检查有限值。图校验失败抛 `ValueError`，不会将错误变成低置信或零分继续执行。OWL 的开放世界语义仍然保留；SHACL 是对这个交换格式指定的校验合同，不是把“没有记录”变成“没有伪造”。

## 7. 检验目标及论文边界

单元测试覆盖缺失/零分区分、logit/margin/probability 区分、局部与整句范围、未标注预测、来源链和循环、矛盾副本、异检测器同分保留、监督字段排除、实际关系导出及可选 SHACL 检查。

研究评估还应验证整条预测流水线的副本注入不变性、关系消融、输入质量改变、未见编辑器/codec、概率校准和拒判。当前关系模型比 flat 增加了部分非线性交互，提升可能来自普通特征扩展；需要随机分族、容量匹配等对照，才能讨论本体结构本身的贡献。当前时间项度量局部绝对偏离，特征名为 `relation/temporal_abs_deviation/{detector_id}`。它是一项非线性时间关系特征，不是结构化序列解码或编辑边界的因果推理。

## 原始标准

- [W3C OWL 2 Primer](https://www.w3.org/TR/owl2-primer/)
- [W3C PROV-O](https://www.w3.org/TR/prov-o/)
- [W3C SOSA/SSN](https://www.w3.org/TR/vocab-ssn/)
- [W3C SHACL](https://www.w3.org/TR/shacl/)
