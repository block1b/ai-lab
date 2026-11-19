# 阶段 2｜Prompt 工程脚手架（结构化输出 + 校验 + 重试）

> 对应打卡表阶段 2 验收：**文本信息抽取器，输出固定 JSON，连续 100 次格式正确率 ≥ 98%**。
> 本脚手架把打卡表里这几项任务落地：结构化 Prompt 模板、Pydantic Schema、few-shot、失败重试、思维链对照。

## 一、文件结构

```
stage2/
├── config.py              # 配置：复用 stage1 的 .env，构建 LLM、统计 token
├── schemas.py             # ★ Pydantic Schema（输出契约 + 校验规则）
├── prompts.py             # 结构化 Prompt 模板（角色/任务/约束/示例/输入）
├── extractor.py           # ★ 抽取器：两种模式 + 错误反馈 + 指数退避重试
├── dataset.py             # 10 条带标注的测试评论
├── cli.py                 # 命令行抽取工具
├── evaluate.py            # ★ 验收脚本：连续 N 次，统计格式正确率
└── examples/
    ├── 01_schema_demo.py  # 纯本地演示 Pydantic 如何拦截错误
    └── 02_with_feedback.py# 真实调用，观察错误反馈重试
```

## 二、快速开始

配置复用 `stage1/.env`，无需重新填 Key。若想单独配置，可复制 `stage1/.env` 为 `stage2/.env`。

```bash
cd stage2
# 先跑纯本地演示（不花钱、不需要 API）
python examples/01_schema_demo.py

# 单次抽取
python cli.py "屏幕很好但电池太差，充电还发烫"

# 手动模式（更透明，能看到校验过程）
python cli.py --mode manual "物流很快，包装仔细"

# 跑验收（先用 20 次试水，再跑 100 次）
python evaluate.py --runs 20
python evaluate.py --runs 100
```

## 三、三个设计要点（对应打卡表任务）

### 1. Pydantic Schema 就是"格式正确"的判定标准

`schemas.py` 里的 `ReviewExtract` 定义了三个字段的**类型 + 值域 + 必填**约束。
模型输出能不能通过 `ReviewExtract.model_validate()`，就是"格式正确率"的机器判定依据——不再靠肉眼。
它还提供 `model_json_schema()`，自动转成 JSON Schema 注入 prompt，让模型看到约束。

### 2. 两种抽取模式，便于对比学习

| 模式 | 原理 | 优点 | 用途 |
|---|---|---|---|
| `structured`（默认） | `with_structured_output(ReviewExtract, method="function_calling")` | 简洁，LangChain 自动注入 Schema + 校验 | 生产用法 |
| `manual` | 让模型吐 JSON 字符串 → 手动解析 → Pydantic 校验 | 每一步透明，能看到原始输出；兼容性最好 | 学习/调试 |

> 兼容性说明：`structured` 默认用 `function_calling`（千问兼容模式普遍支持），初始化时会依次尝试 `json_schema`、默认方式作为回退。若你的模型/端点完全不支持结构化输出，直接加 `--mode manual` 即可——它只依赖普通文本输出 + 本地 Pydantic 校验，任何模型都能跑。

### 3. 失败重试 = 校验异常 + 错误反馈 + 指数退避

`extractor.py` 的重试循环做三件事：
- **捕获** `ValidationError` / `JSONDecodeError` 作为失败信号
- **反馈**：重试时把"上次哪里错了 + 你的原始输出"塞回 prompt，让模型修正（这是关键，比盲目重试有效得多）
- **退避**：等待 0.5 → 1 → 2 → 4 秒（带随机抖动），避免触发限流

### 4. Token 统计为什么用 callback

`with_structured_output` 直接返回 Pydantic 对象、拿不到 response，因此 token 用量通过 `TokenCounter` callback 在 LLM 调用结束时截获（`extractor.py` 顶部）。它同时兼容 LangChain 1.x 的 `usage_metadata` 和 OpenAI 兼容端点的 `response_metadata.token_usage`。无论校验成功或失败，模型那次调用都已计费，所以在 `finally` 里累加——**失败重试的成本也算进去了**。`evaluate.py` 会分别报告输入/输出 token 与单次均值。

## 四、怎么看验收结果

`evaluate.py` 输出三个核心数字：

- **格式正确率**：通过 Pydantic 校验的比例 → **目标 ≥ 98%**（这是硬指标）
- **首次通过率**：第一次就通过、无需重试的比例 → 反映 prompt 质量，越高越好
- **累计 token / 耗时**：成本与延迟感知

如果格式正确率不达标，按这个顺序排查：
1. 看失败样例的原始输出（用 `cli.py --mode manual` 复现）——是 JSON 格式错还是字段值不合规？
2. 值不合规 → 加强 prompt 的约束描述、补 few-shot 示例
3. 格式错（多 Markdown 包裹）→ `extractor.py` 的 `_clean_json` 已处理常见情况，仍有则补充清洗规则
4. 仍不稳 → 调大 `--retries`，或换更强的模型

## 五、成本提醒

`evaluate.py --runs 100` 会真实调用 API 约 100 次（含重试可能更多）。用千问免费额度足够跑；
建议先 `--runs 20 --delay 0.2` 试水，确认正常再跑满 100 次。

## 六、验收自检清单

- [ ] `python examples/01_schema_demo.py` 看到 4 种错误都被 Pydantic 拦截
- [ ] `python cli.py "..."` 能输出情感/属性/证据三个字段
- [ ] `--mode structured` 和 `--mode manual` 都能跑通
- [ ] `python evaluate.py --runs 100` 格式正确率 ≥ 98%
- [ ] 能说清"错误反馈重试"比"盲目重试"好在哪里
