# 基于大模型的移动应用合规检测项目 README

## 1. 项目简介

本项目实现了一个**基于大语言模型（LLM）与检索增强生成（RAG）的移动应用合规检测系统**，面向移动应用隐私合规场景，对应用简介、隐私政策、APK 权限声明以及国家标准文本进行联合分析，自动生成多维度合规分析报告，并提供分类评估、RAG 检索评估、报告质量评估、效率评估与消融实验等完整实验链路。

从代码结构与功能实现来看，该项目并非单一模型调用脚本，而是一个较完整的**“知识库构建—应用分类—多任务合规分析—自动报告生成—实验评估”**研究型原型系统，具有较强的毕业设计与论文实验支撑价值。

---

## 2. 项目目标

项目围绕移动应用隐私合规检测，主要解决以下三个核心问题：

1. **应用分类识别**：根据应用简介与隐私政策，判定应用是否属于 GB/T 41391-2022 涵盖的重点类别，并进一步识别具体类别。
2. **隐私政策合规审查**：结合国标条款，判断隐私政策在告知、同意、敏感信息处理、第三方共享等方面是否满足要求。
3. **权限合规与一致性分析**：从 APK 中提取权限并与隐私政策、国标必要信息范围进行比对，发现“调用未披露”“披露不充分”“超范围申请”等问题。

最终，系统输出三类分析报告：

- 国标和隐私政策分析报告
- 隐私政策和权限分析报告
- 国标和权限分析报告

---

## 3. 技术路线概述

本项目的整体技术路线可以概括为以下五个阶段：

### 3.1 标准知识库构建

针对国家标准 PDF 文档，项目采用如下处理链路：

- `PDF渲染.py`：将 PDF 按页渲染为高清图像；
- `PDF解析.py`：调用 PaddleOCRVL 对 PDF 进行版面解析与 Markdown/JSON 提取；
- `json文本清洗.py`：对解析结果进行结构化清洗，保留章节、表格、附录、上下文语义；
- `chunks清洗.py`：将标准文档切分为适合检索的 chunk，并对附录表格、类别、权限、敏感信息等内容进行强化建模；
- `embedding文本生成.py`：为每个 chunk 生成适用于嵌入模型的文本表示；
- `向量化.py`：调用嵌入模型生成向量，并写入 FAISS 向量索引，形成 `rag_store`。

### 3.2 RAG 检索模块

项目通过 `demo.py` 提供统一的检索接口，核心特征包括：

- 基于 FAISS 的向量检索；
- 结合 chunk 元数据的类型偏置检索；
- 支持 `definition / permission / profile / general` 等不同检索意图；
- 支持 embedding cache，降低重复调用成本；
- 可作为下游应用分类与合规分析脚本的统一知识召回底座。

### 3.3 应用分类模块

`app分类.py` 负责完成应用类别判定，采用两阶段策略：

- **阶段一**：粗粒度判断应用是否属于 GB/T 41391-2022 覆盖的 39 类之一；
- **阶段二**：若属于 39 类，则进一步判定具体类别；若不属于，则生成替代类别解释；
- 在判定过程中，脚本结合应用简介、隐私政策摘要与 RAG 检索出的国标定义/判定方法条款进行约束推理。

### 3.4 多任务合规分析模块

项目围绕三类关系开展自动分析：

#### （1）国标 ↔ 隐私政策
`国标和隐私政策分析.py`

- 面向隐私政策文本执行国标驱动的控制点合规审查；
- 控制点覆盖处理主体、收集必要性、敏感信息单独同意、第三方共享、未成年人规则等；
- 输出结构化 Markdown 报告。

#### （2）隐私政策 ↔ APK 权限
`隐私政策和权限分析.py`

- 从映射后的权限列表出发，分析隐私政策是否对相关能力/数据处理进行了充分披露；
- 采用“规则召回 + LLM 抽取 + 原文匹配 + LLM 判定”的混合策略；
- 支持系统权限与厂商/第三方权限的弱语义推断；
- 输出权限-政策一致性报告。

#### （3）国标 ↔ APK 权限
`国标和权限分析.py`

- 将应用分类结果、权限映射结果、隐私政策证据与国标检索结果联合起来；
- 对每项权限给出“业务核心必要 / 业务辅助合理 / 国标未直接枚举但可解释 / 业务弱相关需审查 / 高风险异常权限”等判定；
- 最终形成权限必要性与风险评估报告。

### 3.5 评估与实验模块

项目不仅提供分析主链路，还提供了较完善的实验评估代码：

- **RAG 检索评估**：`评估/rag评估/rag评估.py`
- **应用分类评估**：`评估/分类评估/分类评估.py`
- **报告质量评估**：`评估/报告质量评估/报告质量评估.py`
- **效率评估**：`评估/效率评估/run_efficiency_eval.py`
- **消融实验**：`评估/消融实验/run_ablation_suite.py`

这一部分对于论文第四章、第五章和第六章的实验组织非常关键，说明项目已具备较好的研究型实验支撑能力。

---

## 4. 核心功能总结

本项目当前代码实现的核心能力包括：

- 国家标准 PDF 的自动渲染、OCR 解析与结构化清洗
- 标准知识库的 chunk 构建与向量化检索
- 面向国标的应用分类识别
- 面向隐私政策的国标合规审查
- 面向 APK 权限的政策一致性检测
- 面向国标必要性范围的权限合理性分析
- 自动生成 Markdown 分析报告
- 面向论文实验的评估、效率分析与消融实验支持

---
## 5. 推荐运行流程
### 第一步：构建标准知识库

#### 1）可选：渲染 PDF
```bash
python PDF渲染.py
```

#### 2）解析 PDF
```bash
python PDF解析.py
```

#### 3）清洗解析后的 JSON
```bash
python json文本清洗.py
```

#### 4）清洗并构建 chunk
```bash
python chunks清洗.py
```

#### 5）生成 embedding 文本
```bash
python embedding文本生成.py
```

#### 6）构建向量库
```bash
python 向量化.py
```

执行完成后，会在 `rag_store/` 下生成：

- `docs.jsonl`
- `idmap.json`
- `faiss.index`
- `stats.json`

### 第二步：应用分类

```bash
python app分类.py
```

输出结果默认写入：

```text
分类/
  应用名.json
```

### 第三步：生成 APK 权限映射

```bash
python apk权限映射.py
```

输出结果默认写入：

```text
dataset/apk映射/
  应用名.txt
```

### 第四步：执行三类合规分析

#### 1）国标和隐私政策分析
```bash
python 国标和隐私政策分析.py
```

#### 2）隐私政策和权限分析
```bash
python 隐私政策和权限分析.py
```

#### 3）国标和权限分析
```bash
python 国标和权限分析.py
```

生成的 Markdown 报告通常写入：

```text
output/
  应用名/
    国标和隐私政策分析结果.md
    隐私政策和权限分析结果.md
    国标和权限分析结果.md
```

---

## 6. 评估与实验使用说明

### 6.1 RAG 检索评估

```bash
python 评估/rag评估/rag评估.py \
  --demo_py demo.py \
  --qa_path 评估/rag评估/rag_eval_dataset_professional_200.jsonl \
  --docs_jsonl rag_store/docs.jsonl
```

该脚本用于评估检索召回质量，适合输出 Recall@K、命中率等指标，支撑论文中 RAG 模块有效性分析。

### 6.2 应用分类评估

```bash
python 评估/分类评估/分类评估.py \
  --gold_csv 评估/分类评估/app_classification.csv \
  --pred_dir 分类 \
  --output_dir 评估/分类评估/classification_eval_outputs
```

输出内容通常包括：

- 二分类准确率
- 多分类准确率/Precision/Recall/F1
- 混淆矩阵
- 错误样本分析

### 6.3 报告质量评估

```bash
python 评估/报告质量评估/报告质量评估.py
```

主要评估：

- 结构完整率
- 证据存在率

适合论文中“报告自动生成质量”部分的量化分析。

### 6.4 效率评估

```bash
python 评估/效率评估/run_efficiency_eval.py --project_dir .
```

该模块通过 `EfficiencyRecorder` 记录：

- 总时延
- RAG 耗时
- LLM 推理耗时
- Token 消耗与估算成本
- 吞吐率

适合论文中“系统运行效率分析”章节。

### 6.5 消融实验

```bash
python 评估/消融实验/run_ablation_suite.py --base_dir . --mode all_with_baseline
```

从 `exp_config.py` 可见，当前消融实验设计包括：

- **baseline**：完整系统
- **e1**：去除 RAG
- **e2**：去除权限语义规范化
- **e3**：替换 embedding 模型
- **e4**：替换 chunk 构建策略
- **e5**：去除 prompt 约束
- **e6**：替换 LLM

---

## 7. 项目亮点

从代码实现质量和论文支撑角度看，本项目有以下亮点：

1. **链路完整**：从标准解析、知识库构建到检测、报告生成、实验评估，形成闭环。
2. **研究性较强**：并非只做工程实现，而是包含分类、检索、报告质量、效率、消融等多组实验。
3. **任务设计合理**：围绕“国标—隐私政策—APK 权限”三类关系构建了多任务分析框架。
4. **RAG 结合实际场景**：不是泛化问答，而是用于国标证据召回和约束推理，应用场景明确。
5. **实验可扩展性较好**：消融实验配置化程度较高，适合继续扩展模型、chunk 和 prompt 策略。

---

## 18. 当前代码中值得注意的问题

### 8.1 API Key 硬编码
多个脚本中直接写入了 `SILICONFLOW_TOKEN` / `API_KEY`。这在实际项目中存在明显安全风险，建议统一改为环境变量读取：

```python
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
```

```bash
export SILICONFLOW_API_KEY=your_key
```

### 8.2 依赖声明不完整
部分脚本依赖 `pandas`、`matplotlib`、`scikit-learn`、`PyMuPDF` 等库，但 `requirements.txt` 中未完全列出，建议补齐。

### 8.3 路径与配置分散
当前不少路径和模型名直接写在脚本顶部。后续建议：

- 将路径配置统一放在 `config.yaml` 或 `.env` 中；
- 将模型选择、topk、阈值等实验超参数集中管理；
- 避免多个脚本内维护重复常量。

### 8.4 模块复用性可进一步提升
项目已具备一定模块化基础，但仍有多个脚本中存在：

- 重复的 LLM 调用逻辑
- 重复的重试逻辑
- 重复的文本清洗/文件读写逻辑

建议后续抽取为：

```text
utils/
  api_client.py
  io_utils.py
  text_utils.py
  rag_utils.py
  report_utils.py
```

### 8.5 面向部署的接口层仍较弱
当前项目主要以批处理脚本为主，更适合作为毕业设计原型和实验系统。若后续想扩展为演示系统，可新增：

- FastAPI 后端
- Web 前端展示页面
- 应用上传与报告下载接口
- 任务队列与日志系统

---
## 9. 启动示例
```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install pandas scikit-learn matplotlib pymupdf urllib3

# 2. 构建知识库
python json文本清洗.py
python chunks清洗.py
python embedding文本生成.py
python 向量化.py

# 3. 应用分类
python app分类.py

# 4. 生成权限映射
python apk权限映射.py

# 5. 执行三类合规分析
python 国标和隐私政策分析.py
python 隐私政策和权限分析.py
python 国标和权限分析.py

# 6. 实验评估
python 评估/分类评估/分类评估.py --gold_csv 评估/分类评估/app_classification.csv --pred_dir 分类 --output_dir 评估/分类评估/classification_eval_outputs
python 评估/报告质量评估/报告质量评估.py
python 评估/效率评估/run_efficiency_eval.py --project_dir .
python 评估/消融实验/run_ablation_suite.py --base_dir . --mode all_with_baseline
```
