# pip install transformers tokenizers tiktoken cachetools pydantic
from typing import List, Dict, Optional, Union
import tiktoken
import logging
from cachetools import LRUCache
from transformers import PreTrainedTokenizerFast
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ===================== 全局配置区（生产可放到yaml配置） =====================
# LRU分词器缓存容量
TOKENIZER_CACHE_MAXSIZE = 16
# 预估安全系数：防止低估输出token造成超预算
PREDICT_SAFETY_FACTOR = 1.3
# 冷启动默认预估token（无历史数据兜底）
DEFAULT_PREDICT_COMPLETION_TOKENS = 256

# 模型别名 -> HF分词器地址
MODEL_TO_TOKENIZER_HF_PATH: Dict[str, str] = {
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3:8b": "meta-llama/Llama-3-8B-Instruct",
    "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

# 模型上下文窗口上限，用于路由预检
MODEL_MAX_CONTEXT_WINDOW: Dict[str, int] = {
    "llama3.1:8b": 128000,
    "llama3:8b": 8192,
    "qwen2:7b": 32768,
    "mistral:7b": 32768,
    "gpt-3.5-turbo": 16384,
    "gpt-4o": 128000,
}

# ===================== 全局缓存 =====================
_TOKENIZER_CACHE = LRUCache(maxsize=TOKENIZER_CACHE_MAXSIZE)
# 输出token历史统计：key=model_name，value=历史中位数completion token，生产替换为Redis/数据库
_MODEL_COMPLETION_STATS: Dict[str, float] = {}

# ===================== 数据模型 =====================
class TokenPredictResult(BaseModel):
    predicted_tokens: int
    strategy: str  # heuristic / history / lightgbm
    safety_factor: float


# ===================== 内部工具：分词器加载缓存 =====================
def _get_cached_fast_tokenizer(model_alias: str) -> Optional[PreTrainedTokenizerFast]:
    """加载并缓存FastTokenizer，失败返回None"""
    if model_alias in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_alias]

    hf_path = MODEL_TO_TOKENIZER_HF_PATH.get(model_alias)
    if not hf_path:
        logger.warning(f"model alias [{model_alias}] missing tokenizer mapping")
        return None

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            hf_path,
            trust_remote_code=True
        )
        _TOKENIZER_CACHE[model_alias] = tokenizer
        return tokenizer
    except Exception as e:
        logger.exception(f"load tokenizer {hf_path} failed", exc_info=e)
        return None


# ===================== 对外API 1：输入消息完整token计数（messages + tools + chat template） =====================
def count_messages_tokens(
    messages: List[Dict[str, str]],
    model: str,
    tools: Optional[List[Dict]] = None
) -> int:
    """
    统计对话完整输入token（包含chat template特殊标记、tool定义）
    使用场景：路由预检、上下文窗口校验、输入成本统计
    """
    if not messages:
        return 0

    # GPT系列使用tiktoken
    if model.startswith("gpt"):
        try:
            enc = tiktoken.encoding_for_model(model)
            concat_text = "\n".join([msg["content"] for msg in messages])
            return len(enc.encode(concat_text))
        except Exception as e:
            logger.exception(f"tiktoken encode failed for model {model}", exc_info=e)
            concat_text = "\n".join([msg["content"] for msg in messages])
            return int(len(concat_text) / 4)

    # 开源模型：hf fast tokenizer + apply_chat_template
    tokenizer = _get_cached_fast_tokenizer(model)
    if tokenizer is None:
        concat_text = "\n".join([msg["content"] for msg in messages])
        return int(len(concat_text) / 4)

    try:
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = tokenizer.encode(full_prompt)
        return len(token_ids)
    except Exception as e:
        logger.exception(f"apply chat template failed model={model}", exc_info=e)
        concat_text = "\n".join([msg["content"] for msg in messages])
        return int(len(concat_text) / 4)


# ===================== 对外API 2：纯文本token计数（模型输出content） =====================
def count_text_tokens(text: str, model: str) -> int:
    """
    统计纯文本token，不套用chat template
    ✅使用场景：vLLM/Ollama不返回usage时，统计真实输出completion token
    """
    if not text:
        return 0

    if model.startswith("gpt"):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception as e:
            logger.exception(f"tiktoken encode text failed model={model}", exc_info=e)
            return int(len(text) / 4)

    tokenizer = _get_cached_fast_tokenizer(model)
    if tokenizer is None:
        return int(len(text) / 4)

    try:
        token_ids = tokenizer.encode(text)
        return len(token_ids)
    except Exception as e:
        logger.exception(f"encode text failed model={model}", exc_info=e)
        return int(len(text) / 4)


# ===================== 对外API 3：路由阶段 预测输出token（调用模型之前） =====================
def predict_completion_tokens(
    model: str,
    input_token_count: int,
    req_max_tokens: Optional[int] = None
) -> TokenPredictResult:
    """
    预估模型输出token数量，用于路由比价、预算控制、上下文预检
    策略优先级：请求传入max_tokens > 历史统计中位数 > 启发式冷启动
    :param model: 模型别名
    :param input_token_count: 已计算好的输入token
    :param req_max_tokens: 请求携带max_tokens上限
    """
    # 优先使用请求自带max_tokens（保守上限）
    if req_max_tokens is not None and req_max_tokens > 0:
        return TokenPredictResult(
            predicted_tokens=int(req_max_tokens),
            strategy="request_max_tokens",
            safety_factor=1.0
        )

    # 策略2：历史统计中位数
    if model in _MODEL_COMPLETION_STATS:
        median_val = _MODEL_COMPLETION_STATS[model]
        pred = int(median_val * PREDICT_SAFETY_FACTOR)
        return TokenPredictResult(
            predicted_tokens=pred,
            strategy="history_stat",
            safety_factor=PREDICT_SAFETY_FACTOR
        )

    # 策略3：冷启动启发式兜底
    return TokenPredictResult(
        predicted_tokens=DEFAULT_PREDICT_COMPLETION_TOKENS,
        strategy="heuristic_cold_start",
        safety_factor=1.0
    )


# ===================== 对外API4：数据闭环更新（每次调用完成后，回填真实输出token） =====================
def update_completion_stats(model: str, real_output_tokens: int):
    """
    生产环境建议：替换为定时任务聚合中位数存入Redis/Clickhouse
    此处为内存简易实现，仅作演示
    """
    # 简易：滑动中位数，正式实现不要用内存存储
    if model not in _MODEL_COMPLETION_STATS:
        _MODEL_COMPLETION_STATS[model] = real_output_tokens
    else:
        # 滑动平均示例，可替换为中位数聚合
        alpha = 0.2
        new_val = (1 - alpha) * _MODEL_COMPLETION_STATS[model] + alpha * real_output_tokens
        _MODEL_COMPLETION_STATS[model] = new_val


# ===================== 对外API5：上下文窗口合法性校验 =====================
def check_context_valid(
    model: str,
    input_tokens: int,
    predicted_output_tokens: int
) -> bool:
    """输入+预估输出 是否小于模型最大上下文窗口"""
    max_win = MODEL_MAX_CONTEXT_WINDOW.get(model)
    if max_win is None:
        logger.warning(f"model {model} missing max context config, skip check")
        return True
    total = input_tokens + predicted_output_tokens
    return total <= max_win

# 先算输入 → 预测输出 → 上下文校验 → 路由选模型 → 请求后端 → 统计真实输出 → 更新统计数据
