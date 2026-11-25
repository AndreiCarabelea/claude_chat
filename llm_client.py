"""LLM client routing and cost tracking for Anthropic, xAI, and DeepSeek models."""

import streamlit as st

MODEL_OPTIONS = {
    "Claude Haiku 5.5": "claude-haiku-5-5",
    "Claude Sonnet 5.5": "claude-sonnet-5-5",
    "Claude Opus 5.5": "claude-opus-5-5"
}

GROK_MODELS = {
    "Grok 4.7": "grok-4.7",
    "Grok 4.3": "grok-4.3",
}

DEEPSEEK_MODELS = {
    "DeepSeek V4.1 Flash 1017": "deepseek-flash",
}

OPENAI_COMPATIBLE_MODELS = {**GROK_MODELS, **DEEPSEEK_MODELS}

ALL_MODEL_OPTIONS = {**MODEL_OPTIONS, **OPENAI_COMPATIBLE_MODELS}

MODEL_PRICING_USD_PER_MILLION = {
    "claude-haiku-5-5": {
        "input": 0.10,
        "output": 0.50,
        "input_over_100k": 0.50,
        "output_over_100k": 2.50,
    },
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "grok-4.7": {"input": 2.00, "output": 6.00},
    "grok-4.3": {"input": 1.25, "output": 2.50},
    "deepseek-flash": {"input": 0.30, "cached_input": 0.006, "output": 1.20},
}


def _extract_usage_from_response(response):
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = int(_usage_field(usage, "input_tokens", 0) or 0)
        output_tokens = int(_usage_field(usage, "output_tokens", 0) or 0)
        # OpenAI/xAI style usage fields
        if not input_tokens:
            input_tokens = int(_usage_field(usage, "prompt_tokens", 0) or 0)
        if not output_tokens:
            output_tokens = int(_usage_field(usage, "completion_tokens", 0) or 0)
        return input_tokens, output_tokens

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            return input_tokens, output_tokens

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {}) or {}
        input_tokens = int(
            token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0
        )
        output_tokens = int(
            token_usage.get(
                "completion_tokens",
                token_usage.get("output_tokens", token_usage.get("reasoning_tokens", 0)),
            )
            or 0
        )

    return input_tokens, output_tokens


def _usage_field(usage, field_name, default=None):
    if isinstance(usage, dict):
        return usage.get(field_name, default)
    return getattr(usage, field_name, default)


def _extract_text_from_response(response):
    """Return visible text blocks, ignoring non-text blocks such as Claude thinking."""
    text_parts = []
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            block_type = block.get("type")
            block_text = block.get("text")
        else:
            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)

        if block_type == "text" and isinstance(block_text, str):
            text_parts.append(block_text)

    if not text_parts:
        raise ValueError("Anthropic response did not contain a text block.")
    return "\n".join(text_parts)


def _build_cost_info(model_name, input_tokens, output_tokens, cached_input_tokens=0):
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model_name)
    info = {
        "model": model_name,
        "input_tokens": int(input_tokens or 0),
        "cached_input_tokens": int(cached_input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "input_price_per_million": None,
        "output_price_per_million": None,
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
    }

    if not pricing:
        return info

    is_over_100k = info["input_tokens"] > 100_000
    input_price = float(
        pricing.get("input_over_100k", pricing["input"])
        if is_over_100k else pricing["input"]
    )
    output_price = float(
        pricing.get("output_over_100k", pricing["output"])
        if is_over_100k else pricing["output"]
    )
    cached_input_tokens = min(info["cached_input_tokens"], info["input_tokens"])
    uncached_input_tokens = info["input_tokens"] - cached_input_tokens
    cached_input_price = float(pricing.get("cached_input", input_price))
    input_cost = (
        (uncached_input_tokens / 1_000_000) * input_price
        + (cached_input_tokens / 1_000_000) * cached_input_price
    )
    output_cost = (info["output_tokens"] / 1_000_000) * output_price

    info["input_price_per_million"] = input_price
    info["output_price_per_million"] = output_price
    info["input_cost"] = input_cost
    info["output_cost"] = output_cost
    info["total_cost"] = input_cost + output_cost
    return info


def _extract_cached_input_tokens(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0

    cached_tokens = _usage_field(usage, "prompt_cache_hit_tokens")
    prompt_details = _usage_field(usage, "prompt_tokens_details")

    if cached_tokens is None:
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens", 0)
        else:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0)
    return int(cached_tokens or 0)


def _cost_info_from_response(model_name, response):
    input_tokens, output_tokens = _extract_usage_from_response(response)
    return _build_cost_info(
        model_name,
        input_tokens,
        output_tokens,
        cached_input_tokens=_extract_cached_input_tokens(response),
    )


def _invoke_openai_compatible_chat(xai_client, model_name, system_text, user_text, max_tokens=2048, temperature=0.75):
    """Call xAI or DeepSeek through an OpenAI-compatible client."""
    response = xai_client.client.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Diagnostic signal for runtime validation
    response_id = getattr(response, "id", None)
    print(
        f"[LLM_DEBUG] model={model_name} "
        f"response_id_type={type(response_id).__name__}"
    )

    text = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        text = getattr(message, "content", "") or ""

    return {
        "text": text,
        "cost_info": _cost_info_from_response(model_name, response),
    }


def _merge_cost_info(total_info, new_info):
    if not new_info:
        return total_info
    if not total_info:
        return dict(new_info)

    merged = dict(total_info)
    merged["input_tokens"] = int(merged.get("input_tokens", 0)) + int(new_info.get("input_tokens", 0))
    merged["cached_input_tokens"] = int(merged.get("cached_input_tokens", 0)) + int(new_info.get("cached_input_tokens", 0))
    merged["output_tokens"] = int(merged.get("output_tokens", 0)) + int(new_info.get("output_tokens", 0))

    input_cost_a = merged.get("input_cost")
    input_cost_b = new_info.get("input_cost")
    output_cost_a = merged.get("output_cost")
    output_cost_b = new_info.get("output_cost")

    merged["input_cost"] = (
        (input_cost_a or 0.0) + (input_cost_b or 0.0)
        if (input_cost_a is not None or input_cost_b is not None)
        else None
    )
    merged["output_cost"] = (
        (output_cost_a or 0.0) + (output_cost_b or 0.0)
        if (output_cost_a is not None or output_cost_b is not None)
        else None
    )
    if merged["input_cost"] is not None or merged["output_cost"] is not None:
        merged["total_cost"] = (merged["input_cost"] or 0.0) + (merged["output_cost"] or 0.0)
    else:
        merged["total_cost"] = None

    return merged


def _format_cost_line(cost_info):
    if not cost_info:
        return "Cost: unavailable"

    model = cost_info.get("model", "unknown")
    input_tokens = int(cost_info.get("input_tokens", 0) or 0)
    output_tokens = int(cost_info.get("output_tokens", 0) or 0)
    total_cost = cost_info.get("total_cost")

    if total_cost is None:
        return f"Cost ({model}): usage {input_tokens} in / {output_tokens} out tokens; pricing unavailable"

    return (
        f"Cost ({model}): ${total_cost:.6f} "
        f"({input_tokens} input + {output_tokens} output tokens)"
    )


def chat_complete(
    anthropic_client,
    xai_client,
    model_name,
    system_text,
    user_text,
    messages=None,
    max_tokens=2048,
    temperature=0.75,
):
    """Route to the correct LLM and return {"text": str, "cost_info": dict}."""
    if model_name in MODEL_OPTIONS.values():
        if not anthropic_client:
            return {"text": "Anthropic client not initialized. Please check your API key.", "cost_info": None}
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=system_text,
            messages=messages or [{"role": "user", "content": user_text}],
        )
        return {
            "text": _extract_text_from_response(response),
            "cost_info": _cost_info_from_response(model_name, response),
        }
    elif model_name in OPENAI_COMPATIBLE_MODELS.values():
        if not xai_client:
            return {"text": "OpenAI-compatible model client not initialized. Please check the provider API key.", "cost_info": None}
        return _invoke_openai_compatible_chat(
            xai_client=xai_client,
            model_name=model_name,
            system_text=system_text,
            user_text=user_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        return {"text": "Selected model not supported.", "cost_info": None}
