# 0.2.0 实际验证记录

日期：2026-11-07。主任务为整段录音真假检测；下列检查证明代码流程可执行，不证明本体融合提升准确性或泛化。

## 已执行

| 检查 | 结果 |
|---|---|
| 完整单元/集成测试 | **102 项全部通过，无跳过**；包含原局部模块与新整段模块 |
| 新整段边界 | 测试证据不参与拟合；组、位置、内容泄漏检查；缺失整位专家时拒判；权重版本/重复别名检查；无标签预测；校准贡献重建概率 |
| 训练恢复 | 模拟中断后恢复，最终权重与连续训练逐张量完全相同（CPU 小型波形测试） |
| 音频格式 | WAV/FLAC；嵌入式 Parquet 行组、Arrow file/stream 的正确行选择与 16 kHz 重采样 |
| RDF/SHACL | 原局部图的正/负例通过；新整段证据图导出与不含标签检查通过；二者词表和验证范围不同 |
| 源/外部索引准备 | ASVspoof2019 LA + InTheWild，共 153,240 条索引，排除 0 条；该全量步骤没有逐条解码音频 |
| 真实音频有界验证 | 32 条录音，两个语料，7 个协议角色；逐条解码并计算波形哈希 |
| 真实专家 | AASIST、LFCC-CNN 各一轮 CPU 训练，随后分别完成 32 条整段推理，共 64 条真实专家观测 |
| 真实融合矩阵 | 5 种融合、关系/质量消融、两种移除专家和两个单专家对照，共 11 组；两个语料分别输出测试指标与 50 次配对 bootstrap |
| 测试规模 | 8 条测试录音（每个语料真假各两条），其余 24 条分属专家训练与 5 个开发角色；仅用于工程验证 |
| 包构建 | 0.2.0 wheel；从独立安装目录导入新模块、读取内置资源和第三方许可证、导出证据图 |

完整测试输出为 `examples/unit_test_log.txt`，去除私有路径的验证摘要为 `examples/utterance_verification.json`。本地完整结果为 `runs/utterance_final_smoke/`，包含真实 checkpoint、缓存、逐录音预测和配置，不随源码上传。

执行命令：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_utterance.py --manifest manifests/utterance_external.jsonl \
  --output runs/utterance_final_smoke --device cpu
python -m build --wheel --no-isolation
```

## 环境

Python 3.10；NumPy 1.26.4；SciPy 1.13.0；scikit-learn 1.7.2；PyTorch 2.6.0+cu118（本次使用 CPU）；soundfile 0.14.0；PyArrow 25.0.1；joblib 1.6.0；rdflib 7.6.0；pyshacl 0.40.2。版本记录见 `requirements-tested.txt`。当前执行环境没有 `/dev/nvidia*`，未运行 GPU 作业。

## 未完成的研究实验

- SSL/fairseq 与真实 XLS-R 权重的训练和推理尚未验证；需要兼容的独立环境。
- 三专家全量训练、三随机种子和完整跨语料评测尚未运行，未产生可用于论文结论的结果。
- 当前真实验证为一轮训练与极小校准/测试集；概率、EER 和置信区间不具备性能判断价值。源码测试通过不能替代收敛和泛化验证。
- 全语料的近重复、跨语料说话人映射、吞吐和显存测量尚未完成。
- 当前数值关系融合不是自动 OWL 推理、GNN 或因果推理；旧版局部功能仍保留，但不是本次主要研究终点。
