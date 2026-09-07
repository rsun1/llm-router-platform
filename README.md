# LLM Router Platform

企业级多模型 LLM 路由平台。根据查询类型、用户等级和 token 数把请求路由到合适的模型，
带配额 / 熔断 / 预算三重策略拦截、失败自动降级、LoRA 适配器管理和监控看板。

> 课程作业项目。核心编排用 LangGraph 实现（`src/llm_router_part7_langgraph.py`）。

---

## 请求流程

```
                                    ┌─────────────────────────┐
START → check_quota → classify → select_model → check_breaker → check_cost
            │                                        │             │
            │                                        │             ↓
            │                                        │       select_adapter
            │                                        │             ↓
            │                                        │        call_model
            │                                        │        ┌────┴────┐
            │                                        │     成功│         │失败
            └────────────────┬───────────────────────┘        │         ↓
                             ↓                                │     fallback
                          reject ───────────────────────────→ record → END
```

三条策略检查（配额 / 熔断 / 预算）任一不过就走 `reject`，但**仍然记录日志**——
不记录的话看板上看不到拦截量。`call_model` 失败后走 `fallback` 降级到同 tier
内更便宜的模型重试一次。

## 目录结构

| 路径 | 内容 |
|---|---|
| `src/llm_router_part1_router.py` | 意图分类、模型选择、兜底模型选择 |
| `src/llm_router_part2_inference.py` | 三家 provider 的调用（openai / anthropic / vllm-ollama）、成本预检 |
| `src/llm_router_part4_monitor.py` | 配额、熔断器、SLA、日志落盘 |
| `src/llm_router_part5_deploy.py` | FastAPI 服务 |
| `src/llm_router_part6_adapters.py` | LoRA 适配器注册 / 灰度 / 转正 / 回滚 |
| `src/llm_router_part7_langgraph.py` | **LangGraph 编排，项目入口** |
| `src/utils/token_tools.py` | 分词、上下文窗口校验、输出 token 预估 |
| `streamlit_ui/app.py` | 七页监控看板 |
| `config/config.yaml` | 模型注册表、路由规则、策略阈值 |

## 运行

### 1. 依赖

```bash
pip install langgraph langchain-core langchain-ollama openai anthropic \
            fastapi streamlit pandas plotly pydantic requests \
            python-dotenv pyyaml tiktoken transformers cachetools
```

### 2. 本地模型（可选，但推荐）

`config.yaml` 里 `provider: vllm` 的两个模型目前走本地 ollama：

```bash
ollama pull mistral
ollama pull llama3.1
```

### 3. API key（可选）

只有路由选到 `gpt-4-turbo` / `claude-3.5-sonnet` 时才需要：

```bash
cp .env.example .env   # 然后填入真实 key
```

### 4. 跑起来

```bash
cd src && python llm_router_part7_langgraph.py   # 跑一次完整流程
streamlit run streamlit_ui/app.py                # 监控看板
```

运行日志写到 `data/call_logs.csv`（首次运行自动创建，已 gitignore）。

---

## 设计说明

<!-- TODO: 以下几节自己补，PLAN.md 第四节"已决"和第七节"实测发现的坑"里的内容直接搬过来就行 -->

### 策略检查为什么放在 classify 之前

<!-- TODO -->

### fallback 的三个设计决策

<!-- TODO: status 记什么 / selected_model 存哪个 / 兜底模型==主模型时怎么办 -->

### 与流程图要求的有意分歧

<!-- TODO: 三处，每处写清楚理由 -->

### 实测数据

<!-- TODO: 三种拒绝路径的延迟、fallback 四条路径、熔断器新旧对比 -->

## 截图

见 `screenshots/`。

## 待完成

- [ ] `compress` 节点（小模型压缩上下文减 token）
- [ ] `POST /route` 端点 + 交互式 UI 页面
- [ ] Streamlit 接真实运行数据（现在读的是样例数据）
- [ ] LoRA 训练与推理接入（需要 GPU）
- [ ] `requirements.txt`、`tests/`
