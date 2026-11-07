# 公开数据导入与实验协议

## 可用实现与边界

`ontoforensics.data` 直接读取本地已下载的 PartialEdit E1/E2 音频、对应 Codec 音频，以及可选 VCTK 真实录音，生成统一 JSONL manifest。导入器不下载大型数据或模型，不修改音频，不自动重采样，也不凭文件名生成声学特征。需要统一音频格式时，另行运行主框架的 `prepare-audio` 命令，见根目录 README。

导入使用标准库读取 PCM、IEEE-float、extensible WAV 和 FLAC 的容器头。`inspect_audio(path)` 返回 `duration_sec`、`sample_rate`、`channels`、`format` 和 `frames`，不依赖 `soundfile`。检查容器头不能替代完整的音频解码检查。

## 先选择正确的数据版本

推荐 [PartialEdit v1.1 官方发布](https://zenodo.org/records/18829689)。作者说明此版本修正时间戳，并新增 `PartialEdit_spk.zip`。论文中的 E3/E4 受到许可限制，没有随公开包发布。不要把公开 E1/E2 的实验描述成四个编辑器实验。

此次实现实际读取了官方 v1.1 的小型 CSV 和 speaker ZIP，并只对两个 Codec 压缩包读取了 64 KB 首部来核对文件名；没有下载完整音频集或模型。核验结果：

- CSV 没有表头，格式为 `filename,(start,end)*N,duration`，时间单位为秒。
- 真实 CSV 有 85,007 行：E1 为 42,471 行，E2 为 42,536 行。它有 4、6、8 列的记录，存在三个编辑区间，不能按 README 的一段/两段描述硬编码。
- 说话人包括 `s5`，不能只匹配 `p` 开头的 ID。
- CSV 只列 E1/E2，不直接列 Codec 版本；各编辑器的条目集合也不应假定完全相同。
- v1.1 CSV 的 MD5 为 `3ff74c4e5cab69013d1d9c4a30f59cae`。导入报告会记录实际 SHA-256，并区分已识别版本与自定义/未核验版本。

最小目录结构如下，允许只下载一个子集：

```text
PartialEdit/
  PartialEdit_E1E2.csv
  PartialEdit_spk.zip
  E1/p237/p237_189_edited_partial_16k.wav
  E1-Codec/p237/p237_189_edited_16k.wav
  E2/p237/p237_189_edited_partial_16k.wav
  E2-Codec/p237/p237_189_edited_16k.wav
```

注意 Codec 文件名少了 `_partial`。适配器仅使用这个已核实的映射，并核对音频时长与区间。若时长差超过 25 ms，或区间超出实际音频，它会拒绝转移标注，不猜测时间平移、拉伸或裁剪。25 ms 只用于比较四舍五入后的总时长，不允许标注超出实际音频。

这个检查确认文件命名关系和时长一致，不能独立证明自定义、另行重采样/剪裁版本仍然逐样本对齐。对这类音频，请提供重新对齐后的标注，不能仅改名冒充官方数据。

**Codec 版本的正标签仍然只覆盖内容编辑区间。**整句经过解码，不意味着整句的内容都被编辑。

## VCTK 真实录音配对

VCTK 0.92 常见路径是 `wav48_silence_trimmed/p225/p225_001_mic1.flac` 或 `..._mic2.flac`；[PyTorch 官方 VCTK loader](https://docs.pytorch.org/audio/stable/_modules/torchaudio/datasets/vctk.html) 明确使用该命名。适配器也支持已整理的 `p225_001.wav`、`p225_001_16k.wav` 和带麦克风后缀的 WAV/FLAC。

将 `bona_fide_root` 指向真实录音目录。存在两种麦克风时，显式传入 `mic="mic1"` 或 `mic="mic2"`。多个候选仍然存在时会报错，不取第一个文件。官方 VCTK 数据存在少量缺失麦克风/录音，不应用另一麦克风静默填补。未带 mic 后缀的预处理音频需要用户自己确认来源。

适配器只加入成功导入编辑样本对应的原录音，一条源录音只加入一次。原录音与编辑后录音的时长可以不同，不能把编辑时间戳搬到原始音频。所有衍生版本和原录音共用 `source_recording_id="vctk:p225_001"`。

## JSONL 字段

一行代表一条录音。必需字段如下：

| 字段 | 类型与含义 |
|---|---|
| `recording_id` | 全局唯一字符串，如 `partialedit:E1:p225_001` |
| `audio_path` | 音频路径；加载时，相对路径以 manifest 所在目录为基准 |
| `source_recording_id` | 原始录音分组，如 `vctk:p225_001`；所有编辑、转码变体保持一致 |
| `speaker_id` | 已知数据集说话人 ID，如 `vctk:p225`；不是模型推测身份 |
| `dataset` | 如 `PartialEdit`、`VCTK` |
| `split` | `train`、`dev`、`calibration`、`test` 或 `unspecified` |
| `duration_sec` | 正的有限秒数，导入时取音频头实际时长 |

其余约定字段如下；导入器会显式生成它们：

| 字段 | 类型与含义 |
|---|---|
| `sample_rate` | 正整数，或未知时为 `null` |
| `label` | `0` 为真实录音，`1` 为含编辑；推理时可省略或为 `null` |
| `fake_intervals` | `[[start,end],...]`，有序、互不重叠的半开区间；`label=1` 时至少一段 |
| `ignore_intervals` | 不参与监督或评测的时间区间，缺省为空 |
| `split_origin` | 划分来源，区分提供的协议与项目生成协议 |
| `editor` | 已知编辑器，仅用于审计/分组 |
| `condition` | 已知处理条件，仅用于审计/分组 |
| `metadata` | 标注版本、CSV 行号、哈希、麦克风、标注转移说明等 |
| `group_id` | 可选的更大分组；存在时也检查跨 split 泄漏 |

标签未知或为 0 时，`fake_intervals` 必须为空。忽略区间独立验证，使用者必须在帧标签、损失和指标中应用它。`label`、`editor`、`condition`、`metadata`、数据集路径和文件名不能被送入预测特征。读取监督信息不等于把已知生成器/codec 真值提供给推理模型。

## Python 接口：真实数据导入

```python
import json
from pathlib import Path
from ontoforensics.data import import_partialedit, validate_manifest, write_manifest

result = import_partialedit(
    root="/data/PartialEdit",
    csv_path="/data/PartialEdit/PartialEdit_E1E2.csv",
    bona_fide_root="/data/VCTK-Corpus-0.92/wav48_silence_trimmed",
    speaker_splits="/data/PartialEdit/PartialEdit_spk.zip",
    mic="mic1",
    subsets=["E1", "E1-Codec"],
    strict=True,
)
validate_manifest(result, check_audio=True)
write_manifest("manifests/partialedit.jsonl", result)
Path("manifests/partialedit.import_report.json").write_text(
    json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

`subsets=None` 会发现本地实际存在的 E1/E1-Codec/E2/E2-Codec 目录，不强迫下载 E2。显式选择的目录缺失时会报错。`include_codec=False` 排除自动发现的 Codec 子集。未提供真实目录时，导入结果可能只有正类，训练端必须检查两个类别是否齐全。

`ImportResult` 是列表子类，遍历和 JSONL 保存方式与普通列表一致，另带 `.report`。默认 `strict=True`，缺文件、非法区间、无法对应的 codec 时长等会立即报错。只有显式 `strict=False` 才跳过问题项，并发出警告；**必须保存完整 `.report`**，其中 `failures` 列出每个失败的路径、原因和 CSV 行号。全量失败可能得到空列表，报告会明确记录，不能把空数据集当成成功实验。

## 官方 split 与开发 calibration 划分

`speaker_splits` 可传官方 ZIP、解压后的目录，或显式 Python 映射。未显式传入时，适配器会在数据根目录寻找 `PartialEdit_spk.zip` 或对应解压目录。它只读取根层：

- `train/train.spk`
- `dev/dev.spk`
- `eval/eval.spk`

ZIP 中还包含 `asvspoof2019/` 下的参考副本，不能误用。`eval` 统一映射成 manifest 的 `test`。未找到协议时保留 `unspecified`，不会悄悄随机划分。训练端应拒绝将 `unspecified` 自动当作训练或测试集。

为本项目单独估计阈值和可靠性，可以运行：

```python
from ontoforensics.data import split_development

records = split_development(result, calibration_fraction=0.5, seed=17)
```

它只按 speaker 将已有 dev 分成 dev/calibration，保留 train/test，保证两个新集合非空，至少需要两个 dev 说话人。同一说话人的所有源录音和变体始终在一起。结果记录种子、比例、原 split 和协议来源；该 calibration 划分属于项目开发协议，**不是官方新设的评测划分**。已存在 calibration 时拒绝再次静默分配。

`validate_manifest` 总是检查录音 ID/路径重复、原录音、说话人及可选大组跨 split 的泄漏。它不能发现没有提供给它的外部训练数据污染；基础模型预训练来源需要另行记录。PartialSpoof 与 PartialEdit 存在 VCTK 来源关联，不能把两者间测试直接称为独立源语料泛化。

## PartialSpoof：格式已核验，当前不声称完成导入

[PartialSpoof v1.2 官方 README](https://zenodo.org/records/5766198/files/README_v1.2?download=1) 给出：

- 音频：`database/{train,dev,eval}/con_wav/*.wav`。
- CM protocol：`SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY`，其中 `KEY` 为 `bonafide` 或 `spoof`。
- VAD 时间标签：`database/vad/{train,dev,eval}/<uttid>.vad`，每行 `start end label`，0 为 spoof、1 为 bona fide、2 为 non-speech。
- `.npy` 分辨率标签以 0 为 spoof、1 为 bona fide，与本项目 `label=1` 表示编辑的约定相反；不能直接照搬。

首个版本实现聚焦 PartialEdit。后续 PartialSpoof 适配器宜优先读取 VAD 文本，明确非语音的忽略规则；不需要加载含 pickle 的 NumPy 字典。仅有 `CON_*` 文件名无法可靠反推出其 VCTK 原始录音，应要求额外来源映射，不能伪造分组键声称已排除跨数据集泄漏。
