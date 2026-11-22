# 阶段 3｜RAG 核心脚手架（检索增强生成）

> 对应打卡表阶段 3 验收：**Recall@5 ≥ 90% · Faithfulness ≥ 90% · 幻觉率 < 5%**。
> 覆盖 RAG 五环节：分块 → 向量化 → 检索 → 重排 → 生成，外加评估闭环。

## 一、为什么这样设计

**轻量依赖即可运行**：向量库默认用 LangChain 内置的 `InMemoryVectorStore`（需 numpy，本环境已装 2.5.5），BM25 用纯 Python 实现（中文 2-gram 分词）。你不需要先装 chromadb/jieba 就能跑通全链路并复现验收指标——先跑通，再优化。密钥复用 `stage1/.env`。

**内置示例语料**：`corpus.py` 里有一份虚构公司的产品手册（15 篇）+ 35 个人工标注测试问题，开箱即用。等你熟悉后，把 `load_documents()` 换成自己的 PDF/Markdown 解析即可。

## 二、文件结构（对应五环节）

```
stage3/
├── config.py            # 配置：LLM + Embeddings（百炼兼容端点）
├── corpus.py            # 示例语料 + 标注测试集
├── chunking.py          # ① 分块：固定 vs 语义
├── vectorstore.py       # ② 向量化 + 建库
├── retrieval.py         # ③ 检索：向量 / BM25 / 混合(RRF)
├── rerank.py            # ④ 重排：LLM-as-judge 精排
├── rag.py               # ⑤ 生成：带引用编号 + 无依据拒答
├── evaluate.py          # ★ 评估闭环：三项硬指标
├── cli.py               # 命令行问答
└── examples/
    ├── 01_chunking_compare.py   # 分块对比实验
    ├── 02_retrieval_compare.py  # 检索对比实验
    └── 03_rerank_compare.py     # 重排对比实验
```

## 三、快速开始

```bash
cd stage3
# 1. 纯本地实验（不花钱，先看分块效果）
python examples/01_chunking_compare.py

# 2. 命令行问答（需 API）
python cli.py "手环的防水等级是多少？"
python cli.py --mode hybrid --rerank "续航多久"

# 3. 评估（先跑召回，不花钱的部分；再加 --faith 跑忠实度）
python evaluate.py --mode hybrid
python evaluate.py --mode hybrid --rerank --faith
```

## 四、三项硬指标怎么算

| 指标 | 计算方式 | 目标 |
|---|---|---|
| **Recall@5** | 前5个检索结果中是否命中人工标注的相关来源（纯计算） | ≥ 90% |
| **Faithfulness** | LLM 裁判判断答案每个事实能否从检索片段找到依据 | ≥ 90% |
| **幻觉率** | 在"文档未提及"类问题上，模型是否编造答案 | < 5% |

## 五、必须亲手做的三个实验（打卡表要求）

1. **分块**：跑 `01`，调 `chunk_size`/`overlap`，再跑 `evaluate.py` 看 Recall 变化，找最优参数
2. **检索**：跑 `02`，对比专有名词（5ATM、蓝牙5.3、客服电话）在向量/BM25/混合下的召回差异
3. **重排**：跑 `03`，看 Top-5 经重排后 Top-3 是否更准；权衡精度提升 vs 多 5 次 LLM 调用的延迟/成本

## 六、关键工程点（RAG 常见坑）

- **专有名词搜不到** → 混合检索（向量语义 + BM25 关键词），本脚手架 `hybrid` 已实现 RRF 融合
- **只取向量 Top-K 精度不够** → 先多取（k*2）再 rerank 精排
- **幻觉编造** → `rag.py` 强制引用编号 + 依据不足时拒答；`is_refusal()` 用于评估
- **分块切碎语义** → 语义分块按句子边界聚合；表格场景需要特殊处理（本示例未含表格，留给你扩展）
- **百炼 embedding 批量限制**（实测踩坑）→ 兼容端点单次最多 10 条文本，`config.py` 的 `BatchingEmbeddings` 已自动分批，否则报 `batch size is invalid, it should not be larger than 10`
- **numpy 依赖** → `InMemoryVectorStore` 的余弦相似度计算需要 numpy（本环境已装 2.5.5），已写入 requirements

## 七、换成你自己的文档

```python
# 在 corpus.py 的 load_documents() 里替换为你自己的 Document 列表：
from langchain_core.documents import Document
Document(page_content="文档正文...", metadata={"source": "文件名.pdf"})
# 同时把 EVAL_QUESTIONS 换成你人工标注的 50 个问题（source 对应 metadata.source）
```

## 八、验收自检清单

- [ ] `python examples/01_chunking_compare.py` 能跑，理解块大小对召回的影响
- [ ] `python cli.py "..."` 能回答并带出引用来源
- [ ] 问"支持无线充电吗？"能正确拒答（文档未提及）
- [ ] `python evaluate.py --mode hybrid` Recall@5 ≥ 90%
- [ ] `python evaluate.py --mode hybrid --faith` 三项指标全部达标
- [ ] 能说清"为什么混合检索比纯向量召回好"、"重排解决了什么问题"
