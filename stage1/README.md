# 阶段 1｜把模型当 API 用熟（LangChain + 通义千问）

> 对应学习计划「阶段 1 · 地基期」：能用 SDK 调用聊天接口，理解 token / 上下文窗口 / 温度，**做出一个输入问题、输出回答并打印 token 消耗的命令行工具**。
> 本阶段**不碰训练/微调**，唯一目标是"把模型当 API 用熟"。

## 一、技术方案

用 `langchain-openai` 的 `ChatOpenAI`，通过阿里云百炼的 **OpenAI 兼容模式**接入千问：

```
你的 Python 代码 ──LangChain──> ChatOpenAI ──HTTPS──> 百炼 compatible-mode/v1 ──> 千问模型
```

为什么不直接用 `ChatTongyi`？因为 OpenAI 兼容模式更稳定、模型名可随时换、后续接任何兼容模型都不用改代码。这是工程上更推荐的做法。

## 二、目录结构

```
stage1/
├── .env.example          # 配置模板（复制为 .env 后填真实 Key）
├── .gitignore            # 排除 .env，防止密钥泄露
├── requirements.txt      # 依赖
├── config.py             # 公共配置：加载 .env、构建 LLM、解析 token
├── cli.py                # ★ 验收工具：命令行问答 + token 统计
└── examples/
    ├── 01_basic.py       # 最简调用
    ├── 02_temperature.py # 温度对比实验
    └── 03_stream.py      # 流式输出 + token 统计
```

## 三、快速开始

### 1. 安装依赖

```bash
cd stage1
pip install -r requirements.txt
```

### 2. 配置密钥

把 `.env.example` 复制为 `.env`，填入你申请到的真实内容：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
DASHSCOPE_API_KEY=sk-你的真实Key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

**模型名怎么填？** 填你账号里实际可用、有免费额度的模型。可用 `qwen-plus` / `qwen-turbo` / `qwen-flash` / `qwen-max`，或你控制台里看到的 `qwen3.8-flash` 等新版本名。以控制台实际可用为准——名字填错会报 `model not found`。

### 3. 运行验收工具

```bash
# 单次提问
python cli.py "什么是 RAG？"

# 自定义系统提示词和温度
python cli.py --system "你是Python老师" --temperature 0.3 "解释装饰器"

# 交互模式（可连续提问，输入 exit 退出）
python cli.py
```

成功时你会看到类似输出：

```
===== 回答 =====
RAG（检索增强生成）……

===== 本次 Token 消耗 =====
  输入 (prompt)     : 28
  输出 (completion) : 96
  合计 (total)      : 124
  模型              : qwen-plus
```

### 4. 跑三个实验

```bash
python examples/01_basic.py
python examples/02_temperature.py
python examples/03_stream.py
```

## 四、三个核心概念（本阶段必须建立直觉）

### ① Token —— 计费和上下文的基本单位

- Token 不是字也不是词，是模型切分文本的单位。中文里大致 1 个字 ≈ 1~2 token，英文 1 个词 ≈ 1~3 token。
- **计费 = 输入 token × 输入单价 + 输出 token × 输出单价**，输出通常比输入贵。
- `cli.py` 打印的 `prompt_tokens` 是"你发给模型的"，`completion_tokens` 是"模型回你的"。
- 动手做：把问题从 10 字换成 1000 字，观察输入 token 怎么涨。

### ② 上下文窗口（context window）—— 一次性最多能塞多少

- 每次调用能处理的 token 总量有上限（输入 + 输出一起算），超出会被截断或报错。
- 千问不同模型窗口不同（常见 128K、1M 等），以控制台模型规格为准。
- 这就是为什么后面做 RAG 要"分块"——文档太长塞不进窗口。
- 动手做：观察 `usage['prompt'] + usage['completion']` 离模型窗口还差多远。

### ③ 温度（temperature）—— 控制随机性

- `temperature=0`：最确定，同样输入几乎同样输出，适合事实问答、信息抽取。
- `temperature` 越高越发散、越有创意，但也越容易不稳定或跑偏。千问常见范围 `[0, 2)`。
- 跑 `02_temperature.py` 对比 0 和 1.5，你会看到明显差异。

## 五、阶段验收自检清单

- [ ] 装好依赖，`python cli.py "你好"` 能正常返回回答
- [ ] 输出里能看到本次的**输入 / 输出 / 合计 token 数**
- [ ] 能通过 `--temperature` 改变回答风格，并说清 0 和 1.5 的区别
- [ ] 能用 `SystemMessage` 设定角色，让回答风格符合预期
- [ ] 知道模型名、密钥放在 `.env` 里，且 `.env` 不会被提交到 Git
- [ ] （加分）跑通流式输出，理解"首 token 延迟"的体感差异

全部勾上，阶段 1 就过关了，可以进入阶段 2（Prompt 工程）。

## 六、常见问题

**Q：报 `model not found` / `InvalidModel`？**
A：`QWEN_MODEL` 填的模型名你账号没权限或名字不对。去百炼控制台「模型列表」复制准确名称。

**Q：报 401 / 鉴权失败？**
A：检查 `.env` 里的 Key 是否正确、有没有多余空格，确认复制的是 `.env` 不是改 `.env.example`。

**Q：能连得上但一直不返回？**
A：确认 `QWEN_BASE_URL` 是 `https://dashscope.aliyuncs.com/compatible-mode/v1`（华北2-北京）。若你开通的是其他地域，换成对应端点。

**Q：token 数一直是 0？**
A：说明响应里没带 `usage`。确认走的是百炼兼容端点；个别模型/接入方式不返回 usage，可在百炼控制台看调用明细。

**Q：免费额度怎么用？**
A：新用户开通百炼通常自动发放每模型约 100 万 token、有效期 90 天。务必在控制台打开「免费额度用完即停」，避免额度耗尽后自动按量付费。
