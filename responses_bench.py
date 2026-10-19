#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
responses_bench.py — OpenAI-compatible /v1/responses benchmark and cache tester.

Pure Python standard library. Commands:
  report  Concurrent TTFT benchmark + cache-persistence HTML report.
  bench   Concurrent/rate-limited benchmark with optional per-request CSV.
  cache   Repeated-prefix cache test, optional cold-gap probe and HTML report.

Latency is reported in three stages so a slow first token can be attributed:
  ① response.created      gateway accepted the request
  ② first delta of any kind   the model actually started emitting
  ③ first visible text delta  what a user perceives as the first token
For reasoning models ③-② is the thinking phase, which would otherwise hide
inside a single TTFT number.

Failures are classified. "budget_exhausted" (reasoning ate max_output_tokens
and left no visible text) is a configuration problem, not a capacity problem —
lowering concurrency will not help it.
"""

import argparse
import csv
import datetime
import hashlib
import html as html_lib
import http.client
import json
import math
import os
import random
import socket
import ssl
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse


_SSL_CTX = None

# ---- 请求画像与进度文件 ----
# 进度文件：设置 BENCH_PROGRESS_FILE 后，各阶段把 {plan, phase, phases{名: {done,total}}} 原子写入该文件
_PROGRESS_PATH = os.environ.get("BENCH_PROGRESS_FILE", "")
_PROGRESS_STATE = {"plan": [], "phase": "", "phases": {}}
_PROGRESS_LOCK = threading.Lock()


def _progress_flush():
    if not _PROGRESS_PATH:
        return
    tmp = _PROGRESS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_PROGRESS_STATE, f, ensure_ascii=False)
    os.replace(tmp, _PROGRESS_PATH)


def progress_plan(plan):
    with _PROGRESS_LOCK:
        _PROGRESS_STATE["plan"] = list(plan)
        _progress_flush()


def progress(phase, done, total):
    with _PROGRESS_LOCK:
        _PROGRESS_STATE["phase"] = phase
        _PROGRESS_STATE["phases"][phase] = {"done": done, "total": total}
        _progress_flush()


def display_url(url):
    """报告里只展示 协议+域名(+端口)+路径；去掉账号密码和查询参数，避免带出密钥。"""
    u = urlparse(url)
    host = u.hostname or ""
    if u.port:
        host += ":%d" % u.port
    return "%s://%s%s" % (u.scheme, host, u.path)


_TLS = threading.local()
_RUN_CACHE_KEY = ""
_RUN_CACHE_BASE = ""

# Deltas belonging to the same TCP flush arrive microseconds apart; a gap below
# this is not a real chunk interval and would drag ITL percentiles to zero.
_FLUSH_GAP = 0.001

_VISIBLE_DELTA_TYPES = (
    "response.output_text.delta",
    "response.refusal.delta",
)

# Providers disagree on the reasoning event name: some providers emit
# reasoning_text while OpenAI and others emit reasoning_summary_text.
_REASONING_DELTA_TYPES = (
    "response.reasoning_text.delta",
    "response.reasoning_summary_text.delta",
)

_ERROR_CLASS_LABELS = (
    ("budget_exhausted", "预算被推理吃满(无正文)"),
    ("empty_response", "空响应(上游无文本事件)"),
    ("stall", "流停滞(长时间无字节)"),
    ("timeout", "整体超时"),
    ("http_429", "HTTP 429 限流"),
    ("http_5xx", "HTTP 5xx"),
    ("http_4xx", "HTTP 4xx"),
    ("api_error", "流内 error / response.failed"),
    ("not_sse", "响应不是 SSE"),
    ("stream_truncated", "流提前结束(无终止事件)"),
    ("stream_broken", "连接中断"),
    ("conn_failed", "连接失败"),
    ("missing", "结果缺失"),
)

# A failure is only evidence of a capacity ceiling if it lands in this set.
_CAPACITY_CLASSES = (
    "stall", "timeout", "http_429", "http_5xx", "stream_broken", "conn_failed",
)
_CONFIG_CLASSES = ("budget_exhausted",)

_PARAGRAPH = (
    "You are a senior software engineer working on a large production codebase. "
    "Follow existing conventions, keep changes minimal, validate assumptions, "
    "handle errors explicitly, and explain the result in concise language. "
    "The following reference context remains stable across this session and is "
    "intentionally repeated so upstream prompt caching can be measured. "
)


def build_prefix(approx_tokens):
    """Build stable text at roughly four characters per token."""
    target_chars = max(1, approx_tokens) * 4
    reps = max(1, target_chars // len(_PARAGRAPH) + 1)
    return (_PARAGRAPH * reps)[:target_chars]


def new_cache_base():
    return "rb-" + uuid.uuid4().hex


def derived_cache_key(label):
    """Create a short, opaque derived key without exposing a user-provided base."""
    seed = "%s|%s|%s" % (
        _RUN_CACHE_KEY or _RUN_CACHE_BASE or new_cache_base(),
        label,
        uuid.uuid4().hex,
    )
    return "rb-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:40]


def unique_cache_key():
    return derived_cache_key("load")


def display_cache_key(value):
    if len(value) <= 16:
        return value
    return value[:10] + "…" + value[-5:]


def normalize_usage(raw):
    """Normalize official Responses usage plus common compatible variants."""
    raw = raw if isinstance(raw, dict) else {}
    in_details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    if not isinstance(in_details, dict):
        in_details = {}
    # Some gateways return the chat-completions shape on /v1/responses.
    out_details = raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}
    if not isinstance(out_details, dict):
        out_details = {}

    def number(*values):
        for value in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return 0

    usage = {
        "input_tokens": number(raw.get("input_tokens"), raw.get("prompt_tokens")),
        "cached_tokens": number(
            in_details.get("cached_tokens"),
            raw.get("cached_tokens"),
            raw.get("prompt_cache_hit_tokens"),
        ),
        "cache_write_tokens": number(
            in_details.get("cache_write_tokens"),
            raw.get("cache_write_tokens"),
        ),
        "output_tokens": number(raw.get("output_tokens"), raw.get("completion_tokens")),
        "reasoning_tokens": number(out_details.get("reasoning_tokens")),
        "total_tokens": number(raw.get("total_tokens")),
    }
    # TPOT must divide the visible-text window by visible tokens only; mixing in
    # reasoning tokens understates it by an order of magnitude.
    usage["visible_tokens"] = max(0, usage["output_tokens"] - usage["reasoning_tokens"])
    # 区分"字段不存在"和"字段是 0"：前者说明上游压根不支持 prompt cache，
    # 后者说明支持但这一发没命中。roll 的判语要靠它分开这两种情况。
    usage["has_cache_field"] = any(
        key in in_details or key in raw
        for key in ("cached_tokens", "prompt_cache_hit_tokens")
    )
    return usage


def make_prompt(args, index, fallback):
    if args.prompt:
        return "%s（第 %d 次）" % (args.prompt, index)
    return fallback % index


def build_payload(args, prefix, prompt, cache_key):
    payload = {
        "model": args.model,
        "instructions": prefix,
        "input": prompt,
        "max_output_tokens": args.max_output_tokens,
        "prompt_cache_key": cache_key,
        "store": False,
        "stream": not args.no_stream,
    }
    # reasoning 有两种形状，真实 Codex 会按上游是不是 OpenAI 系发不同的那种：
    #   OpenAI 系   {"effort":"low","context":"all_turns"}
    #   其它上游    {"summary":"auto"}          ← 注意：没有 effort
    # summary 决定推理要不要流式吐出来。抓包实测有的上游一次请求吐了
    # 上千个 reasoning_text.delta（推理占输出近一半）——
    # 不发 summary 的话上游可能压根不吐推理，量到的 TTFT/TPOT 就和真实场景对不上。
    # 另外部分上游对 effort 有白名单，而真实 Codex 打它根本不发 effort，
    # 硬发反而制造出真实场景不存在的失败。所以两个字段各自独立、都可不发。
    reasoning = {}
    if args.reasoning_effort:
        reasoning["effort"] = args.reasoning_effort
    if args.reasoning_summary:
        reasoning["summary"] = args.reasoning_summary
    if args.reasoning_context:
        reasoning["context"] = args.reasoning_context
    if reasoning:
        payload["reasoning"] = reasoning
    if args.include_encrypted_reasoning:
        payload["include"] = ["reasoning.encrypted_content"]
    if args.cache_retention:
        payload["prompt_cache_retention"] = args.cache_retention
    return payload


def estimate_input_tokens(payload):
    """按请求体字节粗估输入 token 数。

    只用来判断"上游报的 input_tokens 是不是离谱地大"，不追求准确。
    3.4 B/token 是从 Codex 抓包里标定的（中英混合、大量 JSON 结构）：
    没有服务端注入的轮次，实测 input_tokens 与体积/3.4 基本吻合。
    """
    try:
        size = len(json.dumps(payload, ensure_ascii=False))
    except Exception:
        return 0
    return int(size / 3.4)


def _drop_conn():
    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    _TLS.conn = None
    _TLS.key = None


def _get_conn(scheme, host, port, timeout):
    key = (scheme, host, port, timeout)
    if getattr(_TLS, "key", None) == key and getattr(_TLS, "conn", None) is not None:
        return _TLS.conn, True
    _drop_conn()
    if scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=_SSL_CTX)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    _TLS.conn = conn
    _TLS.key = key
    return conn, False


def _error_text(obj):
    if not isinstance(obj, dict):
        return str(obj)[:300]
    err = obj.get("error")
    if not isinstance(err, dict):
        response = obj.get("response")
        if isinstance(response, dict):
            err = response.get("error")
    if isinstance(err, dict):
        kind = err.get("type") or err.get("code") or "error"
        message = err.get("message") or err.get("code") or ""
        return "%s: %s" % (kind, str(message)[:240])
    return str(err or obj.get("message") or "unknown stream error")[:300]


def _fail(cls, message, start=None, retries=0):
    result = {"error": message, "error_class": cls, "retries": retries}
    if start is not None:
        result["t_sent_epoch"] = start
    return result


def _no_text_failure(usage):
    """Separate 'reasoning ate the budget' from 'upstream sent no text at all'."""
    if usage.get("reasoning_tokens"):
        return (
            "budget_exhausted",
            "预算被推理吃满：reasoning=%d tok 用尽 max_output_tokens 后没有正文，"
            "请调大 --max-output-tokens（这是配置问题，降并发无效）"
            % usage["reasoning_tokens"],
        )
    return ("empty_response", "空响应（终止事件无可见文本输出）")


def _collect_output_text(obj):
    """Pull visible text out of a non-streamed Responses body."""
    parts = []
    for item in obj.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "refusal"):
                parts.append(part.get("text") or part.get("refusal") or "")
    return "".join(parts)


def _http_error_class(status):
    if status == 429:
        return "http_429"
    if status >= 500:
        return "http_5xx"
    return "http_4xx"


def send_stream(url, token, payload, timeout=180, stall_timeout=0):
    """Send one Responses request and return staged latency, usage, and text."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
    streaming = bool(payload.get("stream"))
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
        "Accept-Encoding": "identity",
        "User-Agent": "responses-bench/1.1",
        # Some Codex-compatible gateways use this as an affinity fallback.
        "session_id": str(payload.get("prompt_cache_key") or ""),
    }
    # A single read blocking longer than this is a stall, not a slow response.
    read_timeout = min(timeout, stall_timeout) if stall_timeout else timeout

    start = time.time()
    response = None
    retries = 0
    for attempt in (0, 1):
        conn, reused = _get_conn(parsed.scheme, parsed.hostname, port, read_timeout)
        try:
            conn.request("POST", path, body=data, headers=headers)
            response = conn.getresponse()
            break
        except (http.client.HTTPException, OSError) as exc:
            _drop_conn()
            if reused and attempt == 0:
                retries = 1
                continue
            return _fail("conn_failed", "%s: %s" % (type(exc).__name__, exc), start, retries)

    if response.status != 200:
        try:
            body = response.read().decode(errors="replace")
        except Exception:
            body = ""
            _drop_conn()
        return _fail(
            _http_error_class(response.status),
            "HTTP %d: %s" % (response.status, body[:500]),
            start,
            retries,
        )

    if not streaming:
        return _read_nonstream(response, start, retries)

    content_type = (response.getheader("Content-Type") or "").lower()
    if "text/event-stream" not in content_type:
        try:
            body = response.read().decode(errors="replace")
        except Exception:
            body = ""
        return _fail(
            "not_sse",
            "响应不是 SSE（Content-Type=%s）: %s" % (content_type or "-", body[:300]),
            start,
            retries,
        )

    t_created = None
    t_first_delta = None
    t_first_text = None
    last_text_at = None
    itls = []
    delta_count = 0
    flush_count = 0
    text_parts = []
    reasoning_event = ""
    usage = normalize_usage({})
    terminal_type = ""
    terminal_at = None
    saw_done = False

    try:
        while True:
            try:
                raw = response.readline()
            except socket.timeout:
                _drop_conn()
                elapsed = time.time() - start
                if stall_timeout and elapsed < timeout:
                    return _fail(
                        "stall",
                        "流停滞：%ds 内没有收到任何字节（已等 %.1fs）" % (stall_timeout, elapsed),
                        start,
                        retries,
                    )
                return _fail("timeout", "整体超时 %ds（流未结束）" % timeout, start, retries)
            if not raw:
                break
            if time.time() - start > timeout:
                _drop_conn()
                return _fail("timeout", "整体超时 %ds（流未结束）" % timeout, start, retries)
            line = raw.decode(errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if payload_text == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type", "")

            if event_type == "error":
                _drop_conn()
                return _fail("api_error", "流内错误 %s" % _error_text(event), start, retries)
            if event_type == "response.failed":
                _drop_conn()
                return _fail("api_error", "响应失败 %s" % _error_text(event), start, retries)

            now = time.time()
            if event_type == "response.created" and t_created is None:
                t_created = now - start

            if event_type in _REASONING_DELTA_TYPES:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    if not reasoning_event:
                        reasoning_event = event_type
                    if t_first_delta is None:
                        t_first_delta = now - start

            if event_type in _VISIBLE_DELTA_TYPES:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    if t_first_delta is None:
                        t_first_delta = now - start
                    if t_first_text is None:
                        t_first_text = now - start
                        flush_count = 1
                    elif last_text_at is not None:
                        gap = now - last_text_at
                        if gap >= _FLUSH_GAP:
                            itls.append(gap)
                            flush_count += 1
                    last_text_at = now
                    delta_count += 1
                    text_parts.append(delta)

            if event_type in ("response.completed", "response.incomplete"):
                terminal_type = event_type
                terminal_at = time.time()
                final_response = event.get("response") or {}
                if isinstance(final_response, dict):
                    usage = normalize_usage(final_response.get("usage") or {})
    except (http.client.HTTPException, OSError) as exc:
        _drop_conn()
        return _fail("stream_broken", "流中断 %s: %s" % (type(exc).__name__, exc), start, retries)

    # Drain to the end of the body so keep-alive survives. Gateways that emit a
    # [DONE] sentinel used to cost a fresh TLS handshake on every request.
    try:
        response.read()
    except Exception:
        _drop_conn()

    if not terminal_type:
        return _fail(
            "stream_truncated",
            "流提前结束（未收到 response.completed/response.incomplete）",
            start,
            retries,
        )
    if t_first_text is None:
        cls, message = _no_text_failure(usage)
        return _fail(cls, message, start, retries)

    total = (terminal_at or time.time()) - start
    visible = usage.get("visible_tokens") or 0
    tpot = None
    if visible > 1:
        tpot = (total - t_first_text) / (visible - 1)
    return {
        "t_created": t_created,
        "t_first_delta": t_first_delta,
        "t_first_text": t_first_text,
        "ttft": t_first_text,
        "total": total,
        "tpot": tpot,
        "itls": itls,
        "delta_count": delta_count,
        "flush_count": flush_count,
        "reasoning_event": reasoning_event,
        "saw_done": saw_done,
        "usage": usage,
        "text": "".join(text_parts),
        "incomplete": terminal_type == "response.incomplete",
        "retries": retries,
        "t_sent_epoch": start,
        "streamed": True,
    }


def _read_nonstream(response, start, retries):
    """Non-streamed mode: only total latency is meaningful."""
    try:
        body = response.read().decode(errors="replace")
    except socket.timeout:
        _drop_conn()
        return _fail("timeout", "非流式读取超时", start, retries)
    except (http.client.HTTPException, OSError) as exc:
        _drop_conn()
        return _fail("stream_broken", "%s: %s" % (type(exc).__name__, exc), start, retries)
    total = time.time() - start
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return _fail("not_sse", "非流式响应不是 JSON: %s" % body[:300], start, retries)
    if obj.get("error"):
        return _fail("api_error", _error_text(obj), start, retries)
    usage = normalize_usage(obj.get("usage") or {})
    text = _collect_output_text(obj)
    if not text:
        cls, message = _no_text_failure(usage)
        return _fail(cls, message, start, retries)
    return {
        "t_created": None,
        "t_first_delta": None,
        "t_first_text": None,
        "ttft": None,
        "total": total,
        "tpot": None,
        "itls": [],
        "delta_count": 0,
        "flush_count": 0,
        "reasoning_event": "",
        "saw_done": False,
        "usage": usage,
        "text": text,
        "incomplete": obj.get("status") == "incomplete",
        "retries": retries,
        "t_sent_epoch": start,
        "streamed": False,
    }


def fire(args, payload):
    result = send_stream(args.url, args.token, payload, args.timeout, args.stall_timeout)
    # 把"我们发了多少"带回去，指纹那边要拿它和上游报的 input_tokens 对账
    if isinstance(result, dict):
        result["est_input_tokens"] = estimate_input_tokens(payload)
    return result


def pct(values, percentile):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stat_line(label, values):
    if not values:
        print("%-16s (无数据)" % label)
        return
    print(
        "%-16s p50=%.1fms  p90=%.1fms  p99=%.1fms  mean=%.1f  min=%.1f  max=%.1f"
        % (
            label,
            pct(values, 50) * 1000,
            pct(values, 90) * 1000,
            pct(values, 99) * 1000,
            statistics.mean(values) * 1000,
            min(values) * 1000,
            max(values) * 1000,
        )
    )


def metric_json(values):
    if not values:
        return None
    return {
        "n": len(values),
        "p50_ms": round(pct(values, 50) * 1000, 2),
        "p90_ms": round(pct(values, 90) * 1000, 2),
        "p99_ms": round(pct(values, 99) * 1000, 2),
        "mean_ms": round(statistics.mean(values) * 1000, 2),
        "min_ms": round(min(values) * 1000, 2),
        "max_ms": round(max(values) * 1000, 2),
    }


def fmt_usage(usage):
    return (
        "input=%s  cached=%s  cache_write=%s  output=%s  reasoning=%s  visible=%s"
        % (
            usage.get("input_tokens", "-"),
            usage.get("cached_tokens", "-"),
            usage.get("cache_write_tokens", "-"),
            usage.get("output_tokens", "-"),
            usage.get("reasoning_tokens", "-"),
            usage.get("visible_tokens", "-"),
        )
    )


def new_fingerprint():
    return {
        "inputs": {},
        "cached": [],
        "reasoning_events": {},
        "done_sentinel": 0,
        "cache_write_seen": 0,
        "nonstream": 0,
        "inflation": [],      # (上游报的 input_tokens, 我们估的) 之比
    }


def feed_fingerprint(fingerprint, result):
    usage = result.get("usage") or {}
    if usage.get("input_tokens"):
        key = usage["input_tokens"]
        fingerprint["inputs"][key] = fingerprint["inputs"].get(key, 0) + 1
    if usage.get("cached_tokens"):
        fingerprint["cached"].append(usage["cached_tokens"])
    if usage.get("cache_write_tokens"):
        fingerprint["cache_write_seen"] += 1
    event = result.get("reasoning_event")
    if event:
        fingerprint["reasoning_events"][event] = fingerprint["reasoning_events"].get(event, 0) + 1
    if result.get("saw_done"):
        fingerprint["done_sentinel"] += 1
    if result.get("streamed") is False:
        fingerprint["nonstream"] += 1
    est = result.get("est_input_tokens") or 0
    if est and usage.get("input_tokens"):
        fingerprint["inflation"].append(usage["input_tokens"] / float(est))


def summarize_fingerprint(fingerprint):
    inputs = fingerprint["inputs"]
    top_input = max(inputs.items(), key=lambda kv: kv[1])[0] if inputs else 0
    block = 0
    for value in fingerprint["cached"]:
        block = math.gcd(block, value)
    inflation = fingerprint.get("inflation") or []
    return {
        "input_tokens": top_input,
        "input_variants": len(inputs),
        "cache_block": block,
        "reasoning_events": sorted(fingerprint["reasoning_events"]),
        "done_sentinel": fingerprint["done_sentinel"] > 0,
        "cache_write_seen": fingerprint["cache_write_seen"] > 0,
        "inflation_max": max(inflation) if inflation else 0.0,
        "inflation_med": statistics.median(inflation) if inflation else 0.0,
    }


def print_fingerprint(fingerprint):
    info = summarize_fingerprint(fingerprint)
    if not info["input_tokens"] and not info["reasoning_events"]:
        return
    print("\n----- 渠道指纹 -----")
    print(
        "同一前缀 input_tokens: %s（%d 种取值）   cached 块粒度(GCD): %s"
        % (info["input_tokens"] or "—", info["input_variants"], info["cache_block"] or "—")
    )
    print("推理事件命名: %s" % (", ".join(info["reasoning_events"]) or "未出现"))
    print(
        "[DONE] 哨兵: %s   cache_write 字段: %s"
        % ("有" if info["done_sentinel"] else "无", "有" if info["cache_write_seen"] else "无")
    )
    print("口径: 同一前缀下 input_tokens 与块粒度一致，说明上游 tokenizer 与缓存实现同源。")

    # 服务端注入检测。抓包实测有一轮请求体 74KB（约 22K token），
    # input_tokens 却报 123,066 —— 5.5 倍，因为那轮模型在服务端跑了 7 次 web 搜索，
    # usage 是内部多轮的累加。撞上这种情况时成本和缓存口径都会失真，必须告警，
    # 否则"按请求体估成本"会严重低估。
    med = info.get("inflation_med") or 0.0
    top = info.get("inflation_max") or 0.0
    if top >= 2.0:
        print(
            "\n⚠️ input_tokens 远超请求体估算（中位 %.1f× / 峰值 %.1f×）"
            % (med, top)
        )
        print("   上游可能有服务端注入（web 搜索、内部多轮），usage 是累加值。")
        print("   → 本轮的成本与缓存口径不能和其它上游直接横比。")
    elif top:
        print("上游注入: 无（input_tokens 与请求体估算吻合，中位 %.1f×）" % med)


def print_error_breakdown(error_classes):
    if not error_classes:
        return
    labels = dict(_ERROR_CLASS_LABELS)
    print("\n----- 失败分类 -----")
    order = [cls for cls, _ in _ERROR_CLASS_LABELS if cls in error_classes]
    order += [cls for cls in sorted(error_classes) if cls not in labels]
    for cls in order:
        if cls in _CONFIG_CLASSES:
            tag = "配置问题"
        elif cls in _CAPACITY_CLASSES:
            tag = "容量问题"
        else:
            tag = "协议/其它"
        print("  %-18s %-24s x%-4d [%s]" % (cls, labels.get(cls, cls), error_classes[cls], tag))
    capacity = sum(error_classes.get(cls, 0) for cls in _CAPACITY_CLASSES)
    if error_classes.get("budget_exhausted"):
        print("  → budget_exhausted 是配置问题：调大 --max-output-tokens，降并发不会让它变好。")
    if not capacity:
        print("  → 没有任何容量类失败，本轮结果不能作为并发上限的证据。")


def collect_bench(results, request_count, wall):
    created, first_delta, first_text, think = [], [], [], []
    totals, tpots, itls, rows = [], [], [], []
    errors, error_classes = [], {}
    output_tokens = visible_tokens = reasoning_tokens = 0
    deltas = flushes = incomplete = retries = 0
    fingerprint = new_fingerprint()

    for index in range(request_count):
        result = results.get(index, {"error": "missing", "error_class": "missing"})
        if "error" in result:
            cls = result.get("error_class", "missing")
            error_classes[cls] = error_classes.get(cls, 0) + 1
            errors.append((cls, result["error"]))
            retries += result.get("retries", 0)
            rows.append({
                "idx": index, "status": "error", "error_class": cls,
                "created_ms": "", "first_delta_ms": "", "ttft_ms": "", "tpot_ms": "",
                "mean_itl_ms": "", "total_ms": "", "input": "", "cached": "",
                "cache_write": "", "output": "", "reasoning": "", "visible": "",
                "error": result["error"],
            })
            continue

        usage = result["usage"]
        feed_fingerprint(fingerprint, result)
        output_tokens += usage.get("output_tokens") or 0
        visible_tokens += usage.get("visible_tokens") or 0
        reasoning_tokens += usage.get("reasoning_tokens") or 0
        deltas += result.get("delta_count") or 0
        flushes += result.get("flush_count") or 0
        incomplete += int(result.get("incomplete", False))
        retries += result.get("retries", 0)
        totals.append(result["total"])
        if result.get("t_created") is not None:
            created.append(result["t_created"])
        if result.get("t_first_delta") is not None:
            first_delta.append(result["t_first_delta"])
        if result.get("t_first_text") is not None:
            first_text.append(result["t_first_text"])
        if result.get("t_first_delta") is not None and result.get("t_first_text") is not None:
            think.append(max(0.0, result["t_first_text"] - result["t_first_delta"]))
        if result["tpot"] is not None:
            tpots.append(result["tpot"])
        itls.extend(result["itls"])
        mean_itl = statistics.mean(result["itls"]) if result["itls"] else None
        rows.append({
            "idx": index,
            "status": "incomplete" if result.get("incomplete") else "ok",
            "error_class": "",
            "created_ms": round(result["t_created"] * 1000, 1) if result.get("t_created") is not None else "",
            "first_delta_ms": round(result["t_first_delta"] * 1000, 1) if result.get("t_first_delta") is not None else "",
            "ttft_ms": round(result["t_first_text"] * 1000, 1) if result.get("t_first_text") is not None else "",
            "tpot_ms": round(result["tpot"] * 1000, 1) if result["tpot"] is not None else "",
            "mean_itl_ms": round(mean_itl * 1000, 1) if mean_itl is not None else "",
            "total_ms": round(result["total"] * 1000, 1),
            "input": usage.get("input_tokens", 0),
            "cached": usage.get("cached_tokens", 0),
            "cache_write": usage.get("cache_write_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "reasoning": usage.get("reasoning_tokens", 0),
            "visible": usage.get("visible_tokens", 0),
            "error": "",
        })

    ok = len(totals)
    return {
        "ok": ok,
        "fail": len(errors),
        "incomplete": incomplete,
        # Truncated answers and reasoning-starved answers share one root cause.
        "budget_victims": error_classes.get("budget_exhausted", 0) + incomplete,
        "capacity_fail": sum(error_classes.get(cls, 0) for cls in _CAPACITY_CLASSES),
        "retries": retries,
        "wall": wall,
        "req_throughput": ok / wall if wall else 0,
        "out_throughput": output_tokens / wall if wall else 0,
        "visible_throughput": visible_tokens / wall if wall else 0,
        "decode_tps": 1.0 / statistics.median(tpots) if tpots else 0,
        "t_created": created,
        "t_first_delta": first_delta,
        "ttft": first_text,
        "think": think,
        "total": totals,
        "tpot": tpots,
        "itl": itls,
        "errors": errors,
        "error_classes": error_classes,
        "rows": rows,
        "output_tokens": output_tokens,
        "visible_tokens": visible_tokens,
        "reasoning_tokens": reasoning_tokens,
        "deltas": deltas,
        "flushes": flushes,
        "fingerprint": fingerprint,
    }


def run_concurrent(args, request_count, concurrency, request_rate, make_payload):
    results = {}
    rate = float("inf") if request_rate == "inf" else float(request_rate)
    started = time.time()
    progress("load", 0, request_count)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for index in range(request_count):
            payload = make_payload(index)
            futures[executor.submit(fire, args, payload)] = index
            if rate != float("inf") and index < request_count - 1:
                time.sleep(random.expovariate(rate))
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            progress("load", len(results), request_count)
    return results, time.time() - started


def print_bench_summary(bench):
    print("\n===== 压测结果 =====")
    print(
        "成功: %d   失败: %d   截断: %d   预算受害(截断+无正文): %d   容量类失败: %d   静默重试: %d   总墙钟: %.2fs"
        % (
            bench["ok"], bench["fail"], bench["incomplete"], bench["budget_victims"],
            bench["capacity_fail"], bench["retries"], bench["wall"],
        )
    )
    print(
        "请求吞吐: %.2f req/s   输出吞吐: %.1f tok/s（其中可见 %.1f tok/s）   推理占输出: %s"
        % (
            bench["req_throughput"], bench["out_throughput"], bench["visible_throughput"],
            ("%.0f%%" % (100.0 * bench["reasoning_tokens"] / bench["output_tokens"]))
            if bench["output_tokens"] else "—",
        )
    )
    print()
    stat_line("①→created", bench["t_created"])
    stat_line("②→首个delta", bench["t_first_delta"])
    stat_line("③→首个正文", bench["ttft"])
    stat_line("推理段②→③", bench["think"])
    stat_line("TPOT/可见tok", bench["tpot"])
    stat_line("ITL(flush间)", bench["itl"])
    stat_line("总耗时", bench["total"])
    if bench["flushes"]:
        print(
            "%-16s 每 flush 平均 %.1f 个 delta（delta %d / flush %d）"
            % ("SSE 粒度", bench["deltas"] / bench["flushes"], bench["deltas"], bench["flushes"])
        )
    print_error_breakdown(bench["error_classes"])
    if bench["errors"]:
        print("\n前几个错误样本:")
        for cls, error in bench["errors"][:5]:
            print("  - [%s] %s" % (cls, error))
    print_fingerprint(bench["fingerprint"])


def bench_json(meta, bench):
    return {
        "meta": meta,
        "counts": {
            "ok": bench["ok"],
            "fail": bench["fail"],
            "incomplete": bench["incomplete"],
            "budget_victims": bench["budget_victims"],
            "capacity_fail": bench["capacity_fail"],
            "retries": bench["retries"],
        },
        "error_classes": bench["error_classes"],
        "throughput": {
            "wall_s": round(bench["wall"], 3),
            "req_per_s": round(bench["req_throughput"], 3),
            "output_tok_per_s": round(bench["out_throughput"], 1),
            "visible_tok_per_s": round(bench["visible_throughput"], 1),
            "decode_tps": round(bench["decode_tps"], 2),
        },
        "latency": {
            "created": metric_json(bench["t_created"]),
            "first_delta": metric_json(bench["t_first_delta"]),
            "first_text": metric_json(bench["ttft"]),
            "think": metric_json(bench["think"]),
            "tpot_visible": metric_json(bench["tpot"]),
            "itl": metric_json(bench["itl"]),
            "total": metric_json(bench["total"]),
        },
        "tokens": {
            "output": bench["output_tokens"],
            "visible": bench["visible_tokens"],
            "reasoning": bench["reasoning_tokens"],
            "deltas": bench["deltas"],
            "flushes": bench["flushes"],
        },
        "fingerprint": summarize_fingerprint(bench["fingerprint"]),
        "rows": bench["rows"],
    }


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


_CSV_FIELDS = [
    "idx", "status", "error_class", "created_ms", "first_delta_ms", "ttft_ms",
    "tpot_ms", "mean_itl_ms", "total_ms", "input", "cached", "cache_write",
    "output", "reasoning", "visible", "error",
]


def run_bench(args):
    progress_plan(["load"])
    prefix = build_prefix(args.prefix_tokens)
    print(
        "[bench] model=%s concurrency=%d requests=%d rate=%s prefix~%d tok "
        "max_out=%d stream=%s cache_key=%s"
        % (
            args.model, args.concurrency, args.requests, args.request_rate,
            args.prefix_tokens, args.max_output_tokens, not args.no_stream,
            display_cache_key(_RUN_CACHE_KEY),
        )
    )

    def payload(index):
        return build_payload(
            args,
            prefix,
            make_prompt(args, index, "用一句话回答：这是第 %d 个 Responses 压测请求。"),
            _RUN_CACHE_KEY,
        )

    results, wall = run_concurrent(
        args, args.requests, args.concurrency, args.request_rate, payload
    )
    bench = collect_bench(results, args.requests, wall)
    print_bench_summary(bench)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(bench["rows"])
        print("\nCSV 已写入:", args.csv)
    if args.json:
        write_json(args.json, bench_json(make_meta(args, args.concurrency), bench))
        print("JSON 已写入:", args.json)


def cache_row(index, result, cold=False, fired_at=None):
    if "error" in result:
        return {
            "idx": index, "status": "error", "error": result["error"],
            "error_class": result.get("error_class", "missing"), "cold": cold,
            "input": 0, "cached": 0, "cache_write": 0, "output": 0,
            "hit": False, "ttft_ms": "", "total_ms": "", "fired_at": fired_at,
        }
    usage = result["usage"]
    return {
        "idx": index,
        "status": "incomplete" if result.get("incomplete") else "ok",
        "error": "",
        "error_class": "",
        "cold": cold,
        "input": usage.get("input_tokens") or 0,
        "cached": usage.get("cached_tokens") or 0,
        "cache_write": usage.get("cache_write_tokens") or 0,
        "output": usage.get("output_tokens") or 0,
        "hit": (usage.get("cached_tokens") or 0) > 0,
        "ttft_ms": round(result["ttft"] * 1000, 1) if result.get("ttft") is not None else "",
        "total_ms": round(result["total"] * 1000, 1),
        "fired_at": fired_at,
    }


def summarize_cache(rows):
    valid = [row for row in rows if row["status"] != "error"]
    errors = len(rows) - len(valid)
    error_classes = {}
    for row in rows:
        if row["status"] == "error":
            cls = row.get("error_class", "missing")
            error_classes[cls] = error_classes.get(cls, 0) + 1
    hits = sum(1 for row in valid if row["hit"])
    total_input = sum(row["input"] for row in valid)
    total_cached = sum(row["cached"] for row in valid)
    writes = sum(1 for row in valid if not row["hit"] and row["cache_write"] > 0)
    plain_misses = sum(1 for row in valid if not row["hit"] and row["cache_write"] == 0)
    return {
        "n": len(valid),
        "nerr": errors,
        "error_classes": error_classes,
        "hits": hits,
        "req_hit_rate": 100.0 * hits / len(valid) if valid else 0,
        # Responses input_tokens already includes cached input tokens.
        "token_hit_rate": 100.0 * total_cached / total_input if total_input else 0,
        "total_input": total_input,
        "total_cached": total_cached,
        "avg_input": total_input / len(valid) if valid else 0,
        "avg_cached": total_cached / len(valid) if valid else 0,
        "writes": writes,
        "plain_misses": plain_misses,
    }


def print_cache_error_samples(rows, limit=5):
    """打印缓存阶段的错误原文 + 发生时刻。

    原先缓存阶段只报「失败 N」和分类计数，拿不到 HTTP body，也看不出错误是零散抖动
    还是某个时刻起连续挂掉（后者是配额/限流闩死，性质完全不同）。实测曾出现
    42/80 失败且全部集中在后半段，只看计数会误判成"缓存不稳"。
    """
    errors = [row for row in rows if row.get("status") == "error"]
    if not errors:
        return
    print("\n----- 缓存阶段错误样本 -----")
    first, last = errors[0], errors[-1]
    span = ""
    if first.get("fired_at") is not None and last.get("fired_at") is not None:
        span = "，分布 %.0fs ~ %.0fs" % (first["fired_at"], last["fired_at"])
    print("共 %d 个%s" % (len(errors), span))
    for row in errors[:limit]:
        at = row.get("fired_at")
        print("  [%s] t=%s  %s"
              % (row.get("error_class", "?"),
                 ("%.0fs" % at) if at is not None else "?",
                 str(row.get("error", ""))[:200]))


def print_cache_summary(summary):
    print("\n===== 缓存命中率 =====")
    print("有效请求: %d   失败: %d" % (summary["n"], summary["nerr"]))
    if summary["n"]:
        print(
            "请求命中率(cached_tokens>0): %d/%d = %.1f%%"
            % (summary["hits"], summary["n"], summary["req_hit_rate"])
        )
        print(
            "token 命中率(cached/input): %d/%d = %.1f%%"
            % (summary["total_cached"], summary["total_input"], summary["token_hit_rate"])
        )
        print(
            "平均 input/req=%.0f   平均 cached/req=%.0f"
            % (summary["avg_input"], summary["avg_cached"])
        )
        if summary["writes"] or summary["plain_misses"]:
            print(
                "未命中构成: 明确写缓存 %d   无 cache_write 字段 %d"
                % (summary["writes"], summary["plain_misses"])
            )
    print_error_breakdown(summary["error_classes"])


def cache_json(meta, rows, summary, gap=None, key_probe=None):
    payload = {
        "meta": meta,
        "summary": {key: value for key, value in summary.items() if key != "error_classes"},
        "error_classes": summary["error_classes"],
        "rows": rows,
    }
    if gap is not None:
        payload["gap_probe"] = gap
    if key_probe is not None:
        payload["key_isolation_probe"] = key_probe
    return payload


_ROLL_FILLER = (
    "补充说明：以上结论来自实测，记录下来以便后续复核，不要在下一轮里重新推导。"
)


def roll_reply(text, target_tokens):
    """把每轮的 assistant 回复垫到指定 token 量。

    每轮只追加二三十个 token 的话，缓存块粒度会把增长吃掉——上游按块成缓存，
    攒不满一个块就看不出变化，序列看着"横盘"，会被误判成"滚动缓存不生效"。
    垫大一点，每轮的增长跨过块边界，趋势才读得出来。
    """
    text = text or "（空）"
    if target_tokens <= 0:
        return text
    need = max(0, target_tokens * 4 - len(text))
    if not need:
        return text
    reps = need // len(_ROLL_FILLER) + 1
    return text + "\n\n" + (_ROLL_FILLER * reps)[:need]


ROLL_WARM_LEAD_S = 30


def roll_warm(args, prefix, verbose=True):
    if not args.warmup:
        return
    if verbose:
        print("预热 %d 次（把固定前缀写进缓存，不计入统计）..." % args.warmup)
    for index in range(args.warmup):
        fire(args, build_payload(
            args, prefix,
            [{"role": "user", "content": "预热 #%d，回复 OK 即可。" % index}],
            _RUN_CACHE_KEY,
        ))
        if args.delay:
            time.sleep(args.delay)
    if verbose:
        print()


def roll_collect(args, warm_thread=None):
    """一轮轮把历史加长，看 cached_tokens 跟不跟着涨。

    和 Anthropic 那边最大的不同：Responses **没有 cache_control 断点**，
    缓存是自动前缀匹配。所以这里不需要"把断点打在已固定的历史末尾"，
    只要保证 instructions 和 prompt_cache_key 全程不变、input 只在末尾追加。
    """
    prefix = build_prefix(args.prefix_tokens)
    print(
        "[roll] model=%s turns=%d warmup=%d prefix~%d tok turn_tokens=%d cache_key=%s"
        % (args.model, args.turns, args.warmup, args.prefix_tokens,
           args.turn_tokens, display_cache_key(_RUN_CACHE_KEY))
    )
    print("说明: instructions 与 prompt_cache_key 全程固定；input 每轮在末尾追加。")
    print("健康表现: cached_tokens 随 input_tokens 一起涨，付费部分保持很小。\n")

    history = []
    rows = []

    if warm_thread is not None:
        warm_thread.join()
    else:
        roll_warm(args, prefix)

    progress("roll", 0, args.turns)
    for turn in range(1, args.turns + 1):
        question = make_prompt(
            args, turn, "第 %d 轮：用一句话回答，这一轮的编号是多少。")
        msgs = history + [{"role": "user", "content": question}]
        result = fire(args, build_payload(args, prefix, msgs, _RUN_CACHE_KEY))
        if "error" in result:
            print("  第%2d轮 错误[%s]: %s"
                  % (turn, result.get("error_class"), result["error"]))
            break
        usage = result["usage"]
        total = usage["input_tokens"]
        read = usage["cached_tokens"]
        rows.append({
            "turn": turn,
            "items": len(msgs),
            "total": total,
            "read": read,
            "paid": max(0, total - read),
            "rate": (read / total) if total else 0.0,
            "has_read_field": usage.get("has_cache_field", read > 0),
        })
        print("  第%2d轮(%2d条) 总输入=%-7s 缓存=%-7s 付费=%-6s 命中=%5.1f%%"
              % (turn, len(msgs), "{:,}".format(total), "{:,}".format(read),
                 "{:,}".format(total - read), rows[-1]["rate"] * 100))
        progress("roll", turn, args.turns)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant",
                        "content": roll_reply(result.get("text"), args.turn_tokens)})
        if args.delay:
            time.sleep(args.delay)
    return rows


def roll_stats(rows):
    """把滚动明细压成一组可判定的指标（口径和 messages_bench.py 的 roll 一致）。"""
    reads = [x["read"] for x in rows]
    totals = [x["total"] for x in rows]

    # 判据是"命中区增长 ÷ 历史增长"，不是"递增轮次占比"。
    # 后者会被缓存块粒度坑：上游按块成缓存，每轮追加不满一个块时序列看着横盘，
    # 实际是健康的。冷启动那几轮也要剔掉，否则捕获率会虚高到 300%+。
    warm = next((i for i, x in enumerate(rows) if x["read"] >= x["total"] * 0.5), None)
    if warm is None or warm >= len(rows) - 1:
        warm = 0
    hist_growth = totals[-1] - totals[warm]
    read_growth = reads[-1] - reads[warm]
    capture = (read_growth / hist_growth) if hist_growth > 0 else 0.0
    sum_total, sum_read = sum(totals), sum(reads)

    # 样本太少时不下结论。n=2 且首轮是冷启动的话，hist_growth 只有几百、
    # read_growth 却是整个大块（上万），捕获率算出来是 2467% 这种数，
    # 判语还会给 "✅ 滚动缓存正常" —— 明明什么都没测出来。
    if len(rows) - warm < 3:
        verdict, level = ("有效轮次只有 %d 轮（剔除冷启动后 %d 轮），样本不足，不下结论"
                          % (len(rows), len(rows) - warm)), "warn"
    elif not any(x["has_read_field"] for x in rows):
        verdict, level = "上游不返回 cached_tokens —— 不支持 prompt cache", "bad"
    elif max(reads) == 0:
        verdict, level = "全程 cached_tokens=0 —— 缓存未生效", "bad"
    elif capture >= 0.5:
        verdict, level = "命中区随对话增长（捕获 %.0f%%），滚动缓存正常" % (capture * 100), "ok"
    elif capture > 0.15:
        verdict, level = "部分增长（捕获 %.0f%%）—— 可能只是缓存块粒度导致的滞后" % (capture * 100), "warn"
    else:
        verdict, level = ("命中区几乎不增长 —— 只有固定前缀在缓存，增长的历史没进去；"
                          "单发命中率看着好看，但长会话成本会随轮数线性上涨"), "warn"

    return {
        "rows": rows, "reads": reads, "totals": totals,
        "warm_turn": rows[warm]["turn"], "hist_growth": hist_growth,
        "read_growth": read_growth, "capture": capture,
        "per_turn": hist_growth / max(1, len(rows) - 1 - warm),
        "sum_total": sum_total, "sum_read": sum_read,
        "saved": (sum_read / sum_total * 100) if sum_total else 0.0,
        "verdict": verdict, "level": level,
    }


def _roll_paper_style():
    return _html_style()


def _roll_chart(stats):
    """双折线：总输入（虚线）vs 命中区（实线）。"""
    rows = stats["rows"]
    W, H, PL, PR, PT, PB = 480, 190, 56, 10, 12, 26
    iw, ih = W - PL - PR, H - PT - PB
    ymax = max(max(stats["totals"]), 1) * 1.08
    n = max(1, len(rows) - 1)

    def xy(i, v):
        return PL + iw * i / n, PT + ih * (1 - v / ymax)

    def poly(vals):
        return " ".join("%.1f,%.1f" % xy(i, v) for i, v in enumerate(vals))

    grid = "".join(
        '<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
        '<text class="yl" x="%d" y="%.1f">%s</text>'
        % (PL, PT + ih * k / 4.0, W - PR, PT + ih * k / 4.0,
           PL - 6, PT + ih * k / 4.0 + 3,
           "{:,}".format(int(ymax * (1 - k / 4.0))))
        for k in range(5))
    step = max(1, len(rows) // 8)
    xlab = "".join('<text class="xl" x="%.1f" y="%d">%d</text>'
                   % (xy(i, 0)[0], H - 8, r["turn"])
                   for i, r in enumerate(rows) if i % step == 0)
    dots = "".join('<circle cx="%.1f" cy="%.1f" r="2.5" fill="#0d7268"/>'
                   % xy(i, v) for i, v in enumerate(stats["reads"]))
    return ('<svg class="chart" viewBox="0 0 %d %d">%s'
            '<polyline fill="none" stroke="#a9a294" stroke-width="1.5" '
            'stroke-dasharray="4 3" points="%s"/>'
            '<polyline class="draw" fill="none" stroke="#0d7268" stroke-width="2" '
            'points="%s"/>%s%s</svg>'
            % (W, H, grid, poly(stats["totals"]), poly(stats["reads"]), dots, xlab))


def write_roll_html(path, meta, stats):
    esc = lambda value: html_lib.escape(str(value))
    rows = stats["rows"]
    last = rows[-1]
    paid = [r["paid"] for r in rows]
    tagcls = {"ok": "ok", "warn": "warn", "bad": "bad"}[stats["level"]]

    cards = [
        # 主数字放"增量命中率"：它回答的是"对话越长，缓存跟不跟得上"，
        # 而不是单发的命中率——后者只有固定前缀在缓存时照样很好看。
        ("增量命中率", "%.0f%%" % (stats["capture"] * 100),
         "第%d轮起 · 历史增长 %s tok" % (stats["warm_turn"],
                                     "{:,}".format(stats["hist_growth"])),
         "cy" if stats["level"] == "ok" else ("bad" if stats["level"] == "bad" else "am")),
        ("累计省下", "%.0f%%" % stats["saved"],
         "%s → 实付 %s tok" % ("{:,}".format(stats["sum_total"]),
                              "{:,}".format(stats["sum_total"] - stats["sum_read"])), "cy"),
        ("末轮命中率", "%.0f%%" % (last["rate"] * 100),
         "第%d轮 · 上下文 %s tok" % (last["turn"], "{:,}".format(last["total"])), ""),
        ("每轮付费", "{:,}".format(int(statistics.median(paid))),
         "tok · 中位数 · 健康时应保持平稳", "am"),
        ("轮数", str(len(rows)),
         "每轮追加约 %.0f tok" % stats["per_turn"], ""),
    ]
    card_html = "".join(
        '<div class="card %s" style="--i:%d"><div class="card-t">%s</div>'
        '<div class="card-v">%s</div><div class="card-s">%s</div></div>'
        % (cls, i, esc(title), esc(value), esc(subtitle))
        for i, (title, value, subtitle, cls) in enumerate(cards))

    body = "".join(
        "<tr><td>第%d轮</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%.0f%%</td></tr>"
        % (r["turn"], r["items"], "{:,}".format(r["total"]),
           "{:,}".format(r["paid"]), "{:,}".format(r["read"]), r["rate"] * 100)
        for r in rows)
    table = ('<table class="pt"><tr><th>轮次</th><th>条数</th><th>总输入</th>'
             '<th>付费</th><th>命中区</th><th>命中率</th></tr>%s</table>' % body)

    lgd = ('<i style="background:#a9a294"></i>总输入(上下文全量)'
           '<i style="background:#0d7268"></i>命中区(缓存读回，不花钱)')
    note = ("判定 · %s<br>关键看<b>付费</b>那一列：健康时它随轮数保持平稳，"
            "总输入却一路上涨。付费跟着涨 = 聊得越久越贵。" % esc(stats["verdict"]))
    sub = "%s · %s · %s" % (esc(meta["url"]), esc(meta["model"]), esc(meta["time"]))
    footer = ("口径：总输入 = usage.input_tokens（OpenAI 口径 <b>含</b> 缓存，"
              "和 Anthropic 的 input_tokens 正好相反）；"
              "命中区 = input_tokens_details.cached_tokens；付费 = 两者之差。<br>"
              "增量命中率 = 命中区增长 ÷ 历史增长（剔除冷启动轮）——"
              "不看单发命中率，因为只有固定前缀在缓存时它照样好看。<br>"
              "Responses 没有 cache_control 断点，缓存是自动前缀匹配，"
              "所以只要 instructions 与 prompt_cache_key 全程不变即可。")

    html = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>%(model)s 滚动缓存</title><style>%(style)s</style><body><div class="wrap">
<div class="head"><h1>%(model)s <span>滚动缓存</span></h1><div class="sub">%(sub)s</div></div>
<div class="rule"></div>
<div class="cards">%(cards)s</div>
<div class="panel"><h2>滚动缓存 · %(n)d 轮真实对话 · 历史逐轮累积
<span class="tag %(tag)s">%(short)s</span></h2>
<div class="rollgrid"><div>%(chart)s<div class="lgd">%(lgd)s</div>
<div class="tp">%(note)s</div></div><div>%(table)s</div></div></div>
<div class="footer">%(footer)s</div>
</div></body></html>
""" % {
        "model": esc(meta["model"]), "sub": sub, "style": _roll_paper_style(),
        "cards": card_html, "n": len(rows), "tag": tagcls,
        "short": esc(stats["verdict"][:24]), "chart": _roll_chart(stats),
        "lgd": lgd, "note": note, "table": table, "footer": footer,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run_roll(args):
    rows = roll_collect(args)
    if len(rows) < 2:
        print("\n有效轮次不足 2，无法判定滚动缓存。")
        return
    s = roll_stats(rows)
    print("\n===== 滚动缓存结论 =====")
    print("轮数: %d   命中区: %s → %s (%+d)"
          % (len(rows), "{:,}".format(s["reads"][0]),
             "{:,}".format(s["reads"][-1]), s["reads"][-1] - s["reads"][0]))
    print("统计口径: 第%d轮起(剔除冷启动)   历史增长 %+d tok (每轮约 %.0f)   命中区捕获 %.0f%%"
          % (s["warm_turn"], s["hist_growth"], s["per_turn"], s["capture"] * 100))
    print("累计总输入: %s   累计命中: %s   省下 %.1f%%"
          % ("{:,}".format(s["sum_total"]), "{:,}".format(s["sum_read"]), s["saved"]))
    icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}[s["level"]]
    print("判定: %s %s" % (icon, s["verdict"]))
    if s["per_turn"] < 200:
        print("提示: 每轮历史只涨了约 %.0f tok，可能小于上游的缓存块粒度。"
              "加大 --turn-tokens 再测一次会更可靠。" % s["per_turn"])

    meta = make_meta(args, 1)
    if args.html:
        write_roll_html(args.html, meta, s)
        print("\nHTML 报告已写入:", args.html)
    if args.json:
        write_json(args.json, {
            "meta": meta,
            "summary": {k: v for k, v in s.items() if k not in ("rows",)},
            "rows": rows,
        })
        print("JSON 已写入:", args.json)


def run_cache(args):
    prefix = build_prefix(args.prefix_tokens)
    print(
        "[cache] model=%s requests=%d concurrency=%d warmup=%d prefix~%d tok "
        "max_out=%d cache_key=%s"
        % (
            args.model, args.rounds, args.concurrency, args.warmup,
            args.prefix_tokens, args.max_output_tokens, display_cache_key(_RUN_CACHE_KEY),
        )
    )
    print("说明: instructions 与 prompt_cache_key 固定；仅末尾 input 变化。")

    def payload(index):
        return build_payload(
            args,
            prefix,
            make_prompt(args, index, "缓存测试请求 #%d，请用一句话回答。"),
            _RUN_CACHE_KEY,
        )

    if args.warmup:
        print("预热 %d 次..." % args.warmup)
        for index in range(args.warmup):
            fire(args, payload(-1 - index))

    indexed = {}
    fingerprint = new_fingerprint()
    if args.concurrency == 1:
        for index in range(args.rounds):
            result = fire(args, payload(index))
            indexed[index] = result
            if "error" in result:
                print("  #%d 错误[%s]: %s" % (index + 1, result.get("error_class"), result["error"]))
            else:
                feed_fingerprint(fingerprint, result)
                if args.rounds <= 20:
                    print("  #%d %s" % (index + 1, fmt_usage(result["usage"])))
            if args.delay:
                time.sleep(args.delay)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(fire, args, payload(index)): index
                for index in range(args.rounds)
            }
            done = 0
            for future in as_completed(futures):
                result = future.result()
                indexed[futures[future]] = result
                if "error" not in result:
                    feed_fingerprint(fingerprint, result)
                done += 1
                if done % 10 == 0 or done == args.rounds:
                    print("  进度: %d/%d" % (done, args.rounds))

    rows = [
        cache_row(index, indexed.get(index, {"error": "missing", "error_class": "missing"}))
        for index in range(args.rounds)
    ]
    summary = summarize_cache(rows)
    print_cache_summary(summary)
    print_fingerprint(fingerprint)

    # 缓存键隔离性探针：同一前缀，换一个全新的 prompt_cache_key 再发一发。
    # 抓包实测有的上游开新会话、key 换了，第一发就命中 99.7% ——
    # 说明它是按【内容前缀】缓存的，prompt_cache_key 形同虚设。
    # 这个属性直接影响网关的亲和路由值不值得做：上游要是根本不看 key，
    # 把同一会话粘到同一渠道的收益就小得多。
    key_probe = None
    if args.key_isolation:
        print("\n----- 缓存键隔离性探针 -----")
        print("同一前缀，换一个全新的 prompt_cache_key 再发一发 ...")
        fresh = derived_cache_key("isolation")
        probe = fire(args, build_payload(
            args, prefix,
            make_prompt(args, 10 ** 6 + 1, "键隔离探针 #%d，请用一句话回答。"),
            fresh,
        ))
        if "error" in probe:
            print("  探针失败[%s]: %s" % (probe.get("error_class"), probe["error"]))
            key_probe = {"error": probe["error"], "error_class": probe.get("error_class")}
        else:
            usage = probe["usage"]
            hit = (usage.get("cached_tokens") or 0) > 0
            rate = usage["cached_tokens"] / usage["input_tokens"] if usage["input_tokens"] else 0.0
            print("  新 key=%s  input=%d cached=%d（%.1f%%）"
                  % (display_cache_key(fresh), usage["input_tokens"],
                     usage["cached_tokens"], rate * 100))
            if hit:
                print("  → 命中：上游按【内容前缀】缓存，忽略 prompt_cache_key。")
                print("     亲和路由收益有限；同前缀的不同会话之间会共享缓存。")
            else:
                print("  → 未命中：上游按 prompt_cache_key 隔离缓存。")
                print("     换会话就要重新预热；亲和路由（同会话粘同渠道）值得做。")
            key_probe = {
                "fresh_key_hit": hit,
                "input_tokens": usage["input_tokens"],
                "cached_tokens": usage["cached_tokens"],
                "hit_rate": rate,
            }

    gap_probe = None
    if args.gap:
        print("\n----- 空窗探针 -----")
        print("静默 %ds 后再发一次，用于判断缓存是否已过期 ..." % args.gap)
        time.sleep(args.gap)
        probe = fire(args, payload(10 ** 6))
        if "error" in probe:
            print("  探针失败[%s]: %s" % (probe.get("error_class"), probe["error"]))
            gap_probe = {"gap_s": args.gap, "error": probe["error"], "error_class": probe.get("error_class")}
        else:
            usage = probe["usage"]
            alive = (usage.get("cached_tokens") or 0) > 0
            print(
                "  静默 %ds 后: input=%d cached=%d → 缓存%s"
                % (args.gap, usage["input_tokens"], usage["cached_tokens"], "仍然存活" if alive else "已失效")
            )
            gap_probe = {
                "gap_s": args.gap,
                "input_tokens": usage["input_tokens"],
                "cached_tokens": usage["cached_tokens"],
                "alive": alive,
            }

    meta = make_meta(args, args.concurrency)
    if args.html:
        write_cache_html(args.html, meta, rows, summary)
        print("\nHTML 报告已写入:", args.html)
    if args.json:
        write_json(args.json, cache_json(meta, rows, summary, gap_probe, key_probe))
        print("JSON 已写入:", args.json)


def run_report(args):
    # 采样发数优先：内部仍按时长驱动定时器，但对外允许直接说要多少发。
    # 循环条件是 elapsed < duration 且 t=0 也发一发，所以 N 发恰好需要 N×interval 秒。
    if getattr(args, "cache_samples", 0) > 0:
        args.cache_duration = args.cache_samples * max(1, args.cache_interval)
    prefix = build_prefix(args.prefix_tokens)
    print(
        "[report] 首字压测(并发%d, %d请求)%s"
        % (args.concurrency, args.requests,
           "" if args.no_cache else " + 缓存持久性(%ds, 每%ds一发)" % (args.cache_duration, args.cache_interval))
    )
    # 进度计划：load → cache（samples>0 才有）→ roll（turns>0 才有）
    progress_plan((["preflight"] if args.preflight else []) + ["load"]
                  + (["cache"] if not args.no_cache else [])
                  + (["roll"] if args.roll_turns > 0 else []))

    def load_payload(index):
        # Unique keys spread the load across gateway affinity buckets/channels.
        return build_payload(
            args,
            prefix,
            make_prompt(args, index, "首字压测请求 #%d，请简短回答。"),
            unique_cache_key(),
        )

    if args.preflight:
        # 开压前单发一次：鉴权/模型名/余额问题别让 N 个并发一起撞
        progress("preflight", 0, 1)
        print("[预检] 单发一次确认端点可用 ...")
        pre = fire(args, load_payload(-1))
        if "error" in pre:
            print("[预检] 失败，已中止（不再发起并发压测）：\n  %s" % pre["error"])
            return
        print("[预检] 通过：首字 %.0fms" % ((pre.get("ttft") or 0) * 1000))
        progress("preflight", 1, 1)

    # 真实上游普遍有 RPM 限额（实测有的上游只有 60），不限速直接甩 100 个并发
    # 会撞一墙 429——量到的不是首字延迟，是限流器的响应速度。
    rate = str(getattr(args, "request_rate", "inf"))
    print("① 首字压测：并发 %d%s ..."
          % (args.concurrency, "" if rate == "inf"
             else "，限速 %s req/s (≈%.0f RPM)" % (rate, float(rate) * 60)))
    load_results, wall = run_concurrent(
        args, args.requests, args.concurrency, rate, load_payload
    )
    bench = collect_bench(load_results, args.requests, wall)

    # roll_collect 读的是 args.turns/warmup/turn_tokens/delay，
    # report 有自己的同名参数（cache 的 warmup 含义不同），所以这里换一组喂进去
    roll_args = argparse.Namespace(**vars(args))
    roll_args.turns = args.roll_turns
    roll_args.warmup = 2
    roll_args.turn_tokens = args.roll_turn_tokens
    roll_args.delay = 0.0
    roll_warm_thread = None

    cache = None
    timeline_rows = []
    if not args.no_cache:
        persistence_prefix = ("[responses-cache-persistence-%s]\n\n" % uuid.uuid4().hex) + prefix
        persistence_key = derived_cache_key("persistence")

        def cache_payload(index):
            return build_payload(
                args,
                persistence_prefix,
                make_prompt(args, index, "缓存持久性请求 #%d，请简短回答。"),
                persistence_key,
            )

        interval = max(1, args.cache_interval)
        print(
            "② 缓存持久性：同步冷启动 1 次，再每 %ds 一发、持续 %ds ..."
            % (interval, args.cache_duration)
        )
        cold = fire(args, cache_payload(-1))
        timeline_rows = [cache_row(-1, cold, cold=True, fired_at=0.0)]

        pending = []
        scheduled_at = time.time()
        if args.roll_turns > 0:
            # ③ 的预热用的是独立于 ② 的前缀，挪到采样窗口最后 30s 里并行跑，省掉进入 ③ 前的等待
            def _warm_bg():
                try:
                    roll_warm(roll_args, prefix, verbose=False)
                except Exception:
                    pass
            roll_warm_thread = threading.Timer(
                max(0.0, args.cache_duration - ROLL_WARM_LEAD_S), _warm_bg)
            roll_warm_thread.daemon = True
            roll_warm_thread.start()
        tick = 0
        max_lag = 0.0
        _cache_total = max(1, int(math.ceil(args.cache_duration / float(interval))))
        _cache_done = [0]
        progress("cache", 0, _cache_total)

        def _cache_cb(_f):
            with _PROGRESS_LOCK:
                _cache_done[0] += 1
                d = _cache_done[0]
            progress("cache", d, _cache_total)

        with ThreadPoolExecutor(max_workers=min(32, max(1, args.concurrency))) as executor:
            while time.time() - scheduled_at < args.cache_duration:
                submitted_at = time.time() - scheduled_at
                _fut = executor.submit(fire, args, cache_payload(tick))
                _fut.add_done_callback(_cache_cb)
                pending.append((tick, submitted_at, _fut))
                tick += 1
                next_at = scheduled_at + tick * interval
                time.sleep(max(0, next_at - time.time()))
            print("   等待 %d 个缓存请求收尾 ..." % len(pending))
            for index, submitted_at, future in pending:
                result = future.result()
                # x axis must use the moment the request actually went out, not the
                # moment it was queued, or a saturated pool skews the timeline.
                sent = result.get("t_sent_epoch")
                fired_at = (sent - scheduled_at) if sent else submitted_at
                max_lag = max(max_lag, fired_at - submitted_at)
                timeline_rows.append(cache_row(index, result, fired_at=fired_at))
        if max_lag > interval:
            print(
                "   ⚠ worker 池饱和：提交到实际发出最大滞后 %.1fs（> 间隔 %ds），"
                "时间轴已按实际发出时刻校正" % (max_lag, interval)
            )

        scored_rows = timeline_rows[1:]
        cache = summarize_cache(scored_rows)
        cache["rows"] = timeline_rows
        cache["duration"] = args.cache_duration
        cache["interval"] = interval
        cache["first_hit"] = next((row["fired_at"] for row in scored_rows if row["hit"]), None)
        cache["max_lag"] = max_lag

    roll = None
    if args.roll_turns > 0:
        print("③ 滚动缓存：%d 轮，历史逐轮累积 ..." % args.roll_turns)
        roll_rows = roll_collect(roll_args, warm_thread=roll_warm_thread)
        if len(roll_rows) >= 2:
            roll = roll_stats(roll_rows)
            icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}[roll["level"]]
            print("   判定: %s %s" % (icon, roll["verdict"]))
        else:
            print("   有效轮次不足 2，③ 板块跳过")

    print_bench_summary(bench)
    if cache:
        print_cache_summary(cache)
        print_cache_error_samples(cache["rows"])
    meta = make_meta(args, args.concurrency)
    write_report_html(args.html, meta, bench, cache, roll)
    print("\nHTML 报告已写入:", args.html)
    if args.json:
        payload = bench_json(meta, bench)
        if cache:
            payload["cache"] = {
                key: value for key, value in cache.items() if key not in ("rows", "error_classes")
            }
            payload["cache"]["error_classes"] = cache["error_classes"]
            payload["cache"]["rows"] = timeline_rows
        if roll:
            payload["roll"] = {k: v for k, v in roll.items() if k != "rows"}
            payload["roll"]["rows"] = roll["rows"]
        write_json(args.json, payload)
        print("JSON 已写入:", args.json)


def make_meta(args, concurrency):
    return {
        "model": args.model,
        "url": display_url(args.url),
        "concurrency": concurrency,
        "max_output_tokens": args.max_output_tokens,
        "prefix_tokens": args.prefix_tokens,
        "reasoning_effort": args.reasoning_effort or "未指定",
        "stream": not args.no_stream,
        "stall_timeout": args.stall_timeout,
        "timeout": args.timeout,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _nice_ceil(value):
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10 ** exponent
    for multiplier in (1, 2, 2.5, 5, 10):
        if value <= multiplier * base:
            return multiplier * base
    return 10 * base


def _fmt_duration(seconds):
    if seconds is None:
        return "—"
    return "%.1fs" % seconds if seconds >= 10 else "%.0fms" % (seconds * 1000)


def _percentile_table(bench):
    # HTML stays deliberately minimal; the staged breakdown, failure classes and
    # channel fingerprint live in the terminal summary and the --json output.
    metrics = [
        ("TTFT 首字", bench["ttft"]),
        ("TPOT/token", bench["tpot"]),
        ("总耗时", bench["total"]),
    ]
    body = []
    for label, values in metrics:
        if not values:
            body.append("<tr><td>%s</td><td colspan=\"6\">—</td></tr>" % label)
            continue
        body.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                label,
                _fmt_duration(pct(values, 50)),
                _fmt_duration(pct(values, 90)),
                _fmt_duration(pct(values, 99)),
                _fmt_duration(statistics.mean(values)),
                _fmt_duration(min(values)),
                _fmt_duration(max(values)),
            )
        )
    return (
        "<table><thead><tr><th>指标</th><th>p50</th><th>p90</th><th>p99</th>"
        "<th>mean</th><th>min</th><th>max</th></tr></thead><tbody>%s</tbody></table>"
        % "".join(body)
    )


def _timeline_svg(cache, width=500, height=230):
    rows = cache["rows"]
    pad_l, pad_r, pad_t, pad_b = 42, 10, 12, 28
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    duration = max(cache["duration"], 1)
    latencies = [
        row["total_ms"] / 1000.0
        for row in rows if isinstance(row.get("total_ms"), (int, float))
    ]
    y_max = _nice_ceil(max(latencies) if latencies else 1)
    bar_w = max(2.0, plot_w / max(len(rows), 1) * 0.6)
    parts = []
    for tick in range(5):
        y = pad_t + plot_h * (1 - tick / 4.0)
        parts.append(
            '<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>'
            % (pad_l, y, pad_l + plot_w, y)
        )
        parts.append(
            '<text x="%g" y="%g" class="yl">%.1f</text>'
            % (pad_l - 6, y + 3, y_max * tick / 4.0)
        )
    for minute in range(0, int(duration // 60) + 1):
        x = pad_l + plot_w * minute * 60.0 / duration
        parts.append('<text x="%g" y="%g" class="xl">%dm</text>' % (x, height - 8, minute))
    for row in rows:
        fired_at = row.get("fired_at") or 0
        x = pad_l + plot_w * fired_at / duration - bar_w / 2
        if row["status"] == "error":
            parts.append(
                '<rect x="%g" y="%g" width="%g" height="6" fill="#b3261e"/>'
                % (x, pad_t + plot_h - 6, bar_w)
            )
            continue
        latency = row["total_ms"] / 1000.0
        bar_h = plot_h * latency / y_max
        if row.get("cold"):
            color = "#8fa8b8"
        elif row["hit"]:
            ratio = float(row["cached"]) / row["input"] if row["input"] else 0
            color = "#b8560f" if ratio >= 0.8 else "#d9a35e"
        else:
            color = "#cec7b6"
        parts.append(
            '<rect x="%g" y="%g" width="%g" height="%g" fill="%s"/>'
            % (x, pad_t + plot_h - bar_h, bar_w, bar_h, color)
        )
    return '<svg viewBox="0 0 %d %d" class="chart">%s</svg>' % (width, height, "".join(parts))


def _html_style():
    """浅色工程纸皮，和 messages_bench.py 的报告同一套。

    两个脚本的报告经常要并排看，样式一致才不用在两种视觉语言之间来回切。
    """
    return """
 :root{
   --bg:#f2efe7; --surface:#fffdf8; --line:#ded8ca; --line2:#c9c2b0;
   --ink:#22252a; --dim:#7b7566; --am:#b8560f; --cy:#0d7268; --bad:#b3261e;
   --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,"Cascadia Code","DejaVu Sans Mono",monospace;
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:12.5px/1.5 var(--mono);
      -webkit-font-smoothing:antialiased;
      background-image:linear-gradient(#e4dfd2 1px,transparent 1px),
                       linear-gradient(90deg,#e4dfd2 1px,transparent 1px);
      background-size:26px 26px;background-position:-1px -1px}
 .wrap{max-width:1080px;margin:0 auto;padding:0 26px 40px}
 .head{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;
       padding:28px 0 18px;flex-wrap:wrap}
 h1{font-size:19px;margin:0;font-weight:600;letter-spacing:-.3px}
 h1 span{color:var(--am)}
 .sub{color:var(--dim);font-size:11px;word-break:break-all;text-align:right}
 .rule{height:3px;background:linear-gradient(90deg,var(--am),var(--cy) 38%,var(--line) 78%)}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
        gap:1px;background:var(--line);border:1px solid var(--line);
        border-top:none;margin:0 0 22px}
 .card{background:var(--surface);padding:15px 17px 14px;position:relative;
       animation:rise .5s cubic-bezier(.2,.7,.3,1) both;animation-delay:calc(var(--i)*65ms)}
 .card::after{content:"";position:absolute;left:0;top:0;width:100%;height:2px;background:var(--line2)}
 .card.am::after{background:var(--am)} .card.cy::after{background:var(--cy)}
 .card.bad::after{background:var(--bad)}
 .card-t{font-size:10.5px;color:var(--dim);letter-spacing:1.5px;margin-bottom:8px}
 .card-v{font-size:29px;font-weight:600;line-height:1;letter-spacing:-1px;
         font-variant-numeric:tabular-nums}
 .card.am .card-v{color:var(--am)} .card.cy .card-v{color:var(--cy)}
 .card.bad .card-v{color:var(--bad)}
 .card-s{font-size:10.5px;color:var(--dim);margin-top:8px}
 @keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .panel{background:var(--surface);border:1px solid var(--line);padding:15px 17px 17px;
        margin-bottom:16px;animation:rise .5s cubic-bezier(.2,.7,.3,1) both;animation-delay:.28s}
 .panel h2{font-size:11.5px;margin:0 0 13px;color:var(--dim);font-weight:500;
           letter-spacing:1.3px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 .rollgrid{display:grid;grid-template-columns:1.05fr .95fr;gap:20px;align-items:start}
 table,table.pt{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 th,td,.pt th,.pt td{padding:5.5px 8px;text-align:right;font-size:11.5px;
                     border-bottom:1px solid var(--line)}
 th:first-child,td:first-child{text-align:left;color:var(--dim)}
 th{color:var(--dim);font-weight:500;font-size:10px;letter-spacing:1px}
 tr:last-child td{border-bottom:none}
 tr:hover td{background:#f7f3e9}
 .tp{font-size:11px;color:var(--dim);margin-top:11px;line-height:1.7}
 .tp b{color:var(--ink);font-weight:600}
 .chart{width:100%;height:auto;display:block;max-height:250px;overflow:visible}
 .grid{stroke:#eae5d8;stroke-width:1}
 .yl{fill:#96907f;font-size:9.5px;text-anchor:end;font-family:var(--mono)}
 .xl{fill:#96907f;font-size:9.5px;text-anchor:middle;font-family:var(--mono)}
 .draw{stroke-dasharray:4000;stroke-dashoffset:4000;animation:dash 1.5s .45s ease-out forwards}
 @keyframes dash{to{stroke-dashoffset:0}}
 .legend,.lgd{font-size:10.5px;color:var(--dim);margin-top:9px;line-height:1.8}
 .legend i,.lgd i{display:inline-block;width:8px;height:8px;margin:0 4px 0 10px;
                  vertical-align:middle}
 .footer{font-size:10.5px;color:var(--dim);margin-top:18px;padding-top:15px;
         border-top:1px solid var(--line);line-height:1.9}
 .footer b{color:var(--ink)}
 .tag{display:inline-block;padding:2px 9px;font-size:10px;letter-spacing:.6px;
      border:1px solid currentColor}
 .tag.ok{color:var(--cy)} .tag.warn{color:var(--am)} .tag.bad{color:var(--bad)}
 .cache-table{margin-top:16px}.bad{color:var(--bad)}
 @media(max-width:880px){.grid2,.rollgrid{grid-template-columns:1fr}.sub{text-align:left}}
"""


def write_report_html(path, meta, bench, cache, roll=None):
    """单页自包含 HTML：卡片 + ①首字压测 + ②缓存持久性 + ③滚动缓存。

    版式和 messages_bench.py 的报告一致，两边结果可以并排比。
    """
    esc = lambda value: html_lib.escape(str(value))
    total_req = bench["ok"] + bench["fail"]
    p50 = _fmt_duration(pct(bench["ttft"], 50)) if bench["ttft"] else "—"
    p90 = _fmt_duration(pct(bench["ttft"], 90)) if bench["ttft"] else "—"

    cards = [
        ("TTFT p50", p50, "并发 %d · p90 %s" % (meta["concurrency"], p90), "am"),
        ("解码速度", "%.0f" % bench["decode_tps"], "tok/s 每流", ""),
        # 别写成「错误 0/50」——分子分母同时出现时，读者会本能地理解成
        # "50 个里成了 0 个"，正好把最好的结果读成最坏的。成功率打头不会被误读。
        ("成功率", "%.0f%%" % (bench["ok"] * 100.0 / total_req if total_req else 0),
         ("%d 次请求 · 零失败" % total_req) if bench["fail"] == 0
         else ("%d 次请求 · 失败 %d 次" % (total_req, bench["fail"])),
         "bad" if bench["fail"] else ""),
    ]
    if cache:
        cards.append(("缓存命中率", "%.0f%%" % cache["req_hit_rate"],
                      "%d/%d 请求 · token 口径 %.0f%%"
                      % (cache["hits"], cache["n"], cache["token_hit_rate"]), "cy"))
    if roll:
        cards.append((
            "增量命中率", "%.0f%%" % (roll["capture"] * 100),
            "%d 轮滚动 · 累计省下 %.0f%%" % (len(roll["rows"]), roll["saved"]),
            "cy" if roll["level"] == "ok" else ("bad" if roll["level"] == "bad" else "am"),
        ))
    card_html = "".join(
        '<div class="card %s" style="--i:%d"><div class="card-t">%s</div>'
        '<div class="card-v">%s</div><div class="card-s">%s</div></div>'
        % (cls, i, esc(title), esc(value), esc(note))
        for i, (title, value, note, cls) in enumerate(cards))

    throughput = ("请求吞吐 <b>%.2f</b> req/s · 输出吞吐 <b>%.0f</b> tok/s · "
                  "总墙钟 <b>%.1fs</b> · 截断 %d"
                  % (bench["req_throughput"], bench["out_throughput"],
                     bench["wall"], bench["incomplete"]))
    cache_panel = ""
    cache_foot = ""
    if cache:
        first_hit = "%.0fs" % cache["first_hit"] if cache["first_hit"] is not None else "无"
        lgd = ('纵轴=延迟(s) 横轴=分钟 ·'
               '<i style="background:#b8560f"></i>命中'
               '<i style="background:#d9a35e"></i>部分'
               '<i style="background:#cec7b6"></i>未命中'
               '<i style="background:#8fa8b8"></i>冷启动'
               '<i style="background:#b3261e"></i>错误 · 请求 %.0f%% / token %.0f%% · 首次命中 %s'
               % (cache["req_hit_rate"], cache["token_hit_rate"], first_hit))
        cache_panel = ('<div class="panel"><h2>② 缓存 · 持久性 · 连续 %ds</h2>%s<div class="legend">%s</div></div>'
                       % (cache["duration"], _timeline_svg(cache), lgd))
        cache_foot = (" · 缓存: 连续 %ds/每 %ds 一发, 前缀~%d tok, 首次命中 %s · "
                      "明确写缓存 %d 次, 无 cache_write 字段 %d 次"
                      % (cache["duration"], cache["interval"], meta["prefix_tokens"],
                         first_hit, cache["writes"], cache["plain_misses"]))
    footer = ("并发 %d × %d 请求, max_output_tokens=%d, %s%s"
              % (meta["concurrency"], total_req, meta["max_output_tokens"],
                 "非流式" if meta.get("no_stream") else "流式", cache_foot))
    sub = "%s · %s · %s" % (esc(meta["url"]), esc(meta["model"]), esc(meta["time"]))

    html = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(model)s 压测报告</title><style>%(style)s</style><body><div class="wrap">
<div class="head"><h1>%(model)s <span>压测报告</span></h1><div class="sub">%(sub)s</div></div>
<div class="rule"></div>
<div class="cards">%(cards)s</div>
<div class="grid2">
 <div class="panel"><h2>① 首字压测 · 并发 %(conc)d</h2>%(ptable)s<div class="tp">%(throughput)s</div></div>
 %(cachepanel)s
</div>
%(rollpanel)s
<div class="footer">%(footer)s</div>
</div></body></html>
""" % {
        "model": esc(meta["model"]), "sub": sub, "style": _html_style(),
        "cards": card_html, "conc": meta["concurrency"],
        "ptable": _percentile_table(bench), "throughput": throughput,
        "cachepanel": cache_panel,
        "footer": footer, "rollpanel": _roll_panel(roll, esc),
    }
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def _roll_panel(roll, esc):
    """③ 滚动缓存板块：双折线 + 明细表 + 判定。roll 为空时整块不渲染。"""
    if not roll:
        return ""
    rows = roll["rows"]
    body = "".join(
        "<tr><td>第%d轮</td><td>%s</td><td>%s</td><td>%s</td><td>%.0f%%</td></tr>"
        % (r["turn"], "{:,}".format(r["total"]), "{:,}".format(r["paid"]),
           "{:,}".format(r["read"]), r["rate"] * 100)
        for r in rows)
    table = ('<table class="pt"><tr><th>轮次</th><th>总输入</th><th>付费</th>'
             '<th>命中区</th><th>命中率</th></tr>%s</table>' % body)
    lgd = ('<i style="background:#a9a294"></i>总输入(上下文全量)'
           '<i style="background:#0d7268"></i>命中区(缓存读回，不花钱)')
    note = ("第%d轮起统计 · 历史增长 %+d tok · 命中区捕获 %.0f%% · "
            "累计省下 %.1f%%（%s → 实付 %s）"
            % (roll["warm_turn"], roll["hist_growth"], roll["capture"] * 100,
               roll["saved"], "{:,}".format(roll["sum_total"]),
               "{:,}".format(roll["sum_total"] - roll["sum_read"])))
    tagcls = {"ok": "ok", "warn": "warn", "bad": "bad"}[roll["level"]]
    # 左图右表：板块是整宽的，图若独占一行会被 width:100% 拉到几倍高，
    # 纵轴刻度被扯得老远、读数反而更费劲。
    return ('<div class="panel"><h2>③ 滚动缓存 · %d 轮真实对话 · 历史逐轮累积'
            '<span class="tag %s">%s</span></h2>'
            '<div class="rollgrid"><div>%s<div class="lgd">%s</div>'
            '<div class="tp">%s</div></div><div>%s</div></div></div>'
            % (len(rows), tagcls, esc(roll["verdict"][:24]),
               _roll_chart(roll), lgd, esc(note), table))


def write_cache_html(path, meta, rows, summary):
    esc = lambda value: html_lib.escape(str(value))
    cards = [
        ("请求命中率", "%.1f%%" % summary["req_hit_rate"], "%d/%d" % (summary["hits"], summary["n"])),
        ("token 命中率", "%.1f%%" % summary["token_hit_rate"],
         "%d/%d tok" % (summary["total_cached"], summary["total_input"])),
        ("有效/失败", "%d / %d" % (summary["n"], summary["nerr"]), "请求数"),
        ("平均 cached", "%.0f" % summary["avg_cached"], "平均 input %.0f" % summary["avg_input"]),
    ]
    card_html = "".join(
        '<div class="card"><div class="card-v">%s</div><div class="card-t">%s</div>'
        '<div class="card-s">%s</div></div>' % (esc(value), esc(title), esc(note))
        for title, value, note in cards
    )
    body = []
    for row in rows:
        body.append(
            "<tr><td>#%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td class=\"bad\">%s</td></tr>"
            % (
                row["idx"] + 1, esc(row["status"]), esc(row["input"]),
                esc(row["cached"]), esc(row["cache_write"]), esc(row["ttft_ms"]),
                esc(row["total_ms"]), esc(row["error"]),
            )
        )
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Responses 缓存报告 - %(model)s</title><style>%(style)s</style></head>
<body><div class="wrap"><div class="head"><h1>Responses API 缓存报告</h1>
<div class="sub">%(url)s · %(model)s · %(time)s</div></div>
<div class="cards" style="grid-template-columns:repeat(4,1fr)">%(cards)s</div>
<section class="panel cache-table"><table><thead><tr><th>请求</th><th>状态</th>
<th>input</th><th>cached</th><th>cache_write</th><th>TTFT(ms)</th>
<th>总耗时(ms)</th><th>错误</th></tr></thead><tbody>%(rows)s</tbody></table></section>
<div class="footer">Responses 的 input_tokens 已包含 cached_tokens；
token 命中率按 cached/input 计算。前缀约 %(prefix)d tokens。</div>
</div></body></html>""" % {
        "model": esc(meta["model"]),
        "style": _html_style(),
        "url": esc(meta["url"]),
        "time": esc(meta["time"]),
        "cards": card_html,
        "rows": "".join(body),
        "prefix": meta["prefix_tokens"],
    }
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return parsed


def validate_args(parser, args):
    parsed = urlparse(args.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        parser.error("--url 必须是完整的 http/https URL")
    if not args.token:
        parser.error("请设置环境变量 BENCH_API_KEY（或 OPENAI_API_KEY），或传 --token")
    if parsed.username or parsed.password:
        parser.error("--url 不能包含用户名或密码")
    if args.prompt_cache_key and len(args.prompt_cache_key) > 64:
        parser.error("--prompt-cache-key 最多 64 个字符")
    if args.stall_timeout and args.stall_timeout > args.timeout:
        parser.error("--stall-timeout 不能大于 --timeout")
    if getattr(args, "request_rate", "inf") != "inf":
        try:
            if float(args.request_rate) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--request-rate 必须是正数或 inf")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible /v1/responses 压测 + 缓存测试"
    )
    parser.add_argument("--url", required=True, help="完整端点，如 https://host/v1/responses")
    parser.add_argument(
        "--token",
        default=os.environ.get("BENCH_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="Bearer token；推荐用环境变量 BENCH_API_KEY（或 OPENAI_API_KEY）传入，避免出现在进程列表",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        default=2048,
        help="reasoning + 可见正文共用的预算；推理模型不要往下调，否则正文会被思考挤掉",
    )
    parser.add_argument("--prefix-tokens", type=positive_int, default=8000)
    parser.add_argument("--timeout", type=positive_int, default=180, help="单请求总超时秒数")
    parser.add_argument(
        "--stall-timeout",
        type=nonnegative_int,
        default=120,
        help="多久没收到任何字节算流停滞（区分“卡死”和“慢”）；0 关闭",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="覆盖内置问题；想量持续解码就要一个长回答，如“写一段500字的介绍”",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="非流式模式；只有总耗时有意义，可用于识别伪流式网关",
    )
    parser.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验")
    parser.add_argument(
        "--prompt-cache-key",
        default="",
        help="固定缓存键（bench/cache 直接使用；report 只作为派生种子）",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="",
        help="可选 reasoning.effort；留空不发送该字段",
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=("auto", "concise", "detailed"),
        default="",
        help="可选 reasoning.summary；留空不发送。真实 Codex 打非 OpenAI 上游"
             "发的就是 summary=auto 且不带 effort，想复刻那个场景就用这个",
    )
    parser.add_argument(
        "--reasoning-context",
        choices=("all_turns",),
        default="",
        help="可选 reasoning.context；真实 Codex 只在 OpenAI 系上游发 all_turns",
    )
    parser.add_argument(
        "--include-encrypted-reasoning",
        action="store_true",
        help="发送 include=[\"reasoning.encrypted_content\"]，三家真实 Codex 都发这个",
    )
    parser.add_argument(
        "--cache-retention",
        choices=("24h",),
        default="",
        help="可选 prompt_cache_retention；模型/网关不支持时请留空",
    )

    commands = parser.add_subparsers(dest="cmd", required=True)

    bench = commands.add_parser("bench", help="并发/限速压测")
    bench.add_argument("--concurrency", type=positive_int, default=10)
    bench.add_argument("--requests", type=positive_int, default=50)
    bench.add_argument("--request-rate", default="inf", help="req/s，或 inf")
    bench.add_argument("--csv", default="")
    bench.add_argument("--json", default="", help="机器可读结果，便于跨端点对比")
    bench.set_defaults(func=run_bench)

    report = commands.add_parser("report", help="一页 HTML：首字压测 + 缓存持久性")
    report.add_argument("--concurrency", type=positive_int, default=100)
    report.add_argument("--requests", type=positive_int, default=100)
    report.add_argument("--cache-duration", type=positive_int, default=180)
    report.add_argument("--cache-interval", type=positive_int, default=8)
    report.add_argument("--cache-samples", type=nonnegative_int, default=40,
                        help="缓存阶段采样发数；给了就按 samples×interval 反推时长，"
                             "覆盖 --cache-duration（省得每次手算）")
    report.add_argument("--preflight", action="store_true",
                        help="开压前单发预检，失败即中止")
    report.add_argument("--no-cache", action="store_true",
                        help="跳过 ② 缓存持久性阶段（快速档用）")
    report.add_argument("--html", default="responses_report.html")
    report.add_argument("--json", default="")
    report.add_argument("--request-rate", default="inf",
                        help="① 首字压测的到达速率上限 req/s (Poisson)；inf=不限速。"
                             "上游有 RPM 限额时必须给，否则满屏 429。RPM 60 → 填 0.9")
    report.add_argument("--roll-turns", type=nonnegative_int, default=12,
                        help="③ 滚动缓存跑多少轮；0 = 不测这一块（默认 12）")
    report.add_argument("--roll-turn-tokens", type=nonnegative_int, default=500,
                        help="滚动测试里每轮 assistant 回复垫到多少 token")
    report.set_defaults(func=run_report)

    cache = commands.add_parser("cache", help="缓存命中率细测")
    cache.add_argument("--rounds", type=positive_int, default=20)
    cache.add_argument("--concurrency", type=positive_int, default=1)
    cache.add_argument("--warmup", type=nonnegative_int, default=2,
                       help="正式统计前先打热几发。缓存写入对后续可见有一发延迟，"
                            "warmup=1 时第 1 发必然 miss，会稀释统计")
    cache.add_argument("--key-isolation", action="store_true",
                       help="跑完后加一发探针：同前缀换全新 prompt_cache_key，"
                            "看上游是按内容前缀缓存还是按 key 隔离")
    cache.add_argument("--delay", type=float, default=0)
    cache.add_argument(
        "--gap",
        type=nonnegative_int,
        default=0,
        help="轮次跑完后静默 N 秒再发一次探针，用于测缓存 TTL 边界",
    )
    cache.add_argument("--html", default="")
    cache.add_argument("--json", default="")
    cache.set_defaults(func=run_cache)

    roll = commands.add_parser(
        "roll", help="滚动缓存：对话越长，缓存跟不跟得上")
    roll.add_argument("--turns", type=positive_int, default=12,
                      help="滚动多少轮（默认 12）")
    roll.add_argument("--warmup", type=nonnegative_int, default=2,
                      help="正式统计前先打热几发，把固定前缀写进缓存")
    roll.add_argument("--turn-tokens", type=nonnegative_int, default=400,
                      help="把每轮的 assistant 回复垫到这么多 token。"
                           "太小的话每轮增量攒不满上游的缓存块，趋势读不出来")
    roll.add_argument("--delay", type=float, default=0.0,
                      help="每轮之间的间隔秒数")
    roll.add_argument("--html", default="", help="导出单页 HTML 报告")
    roll.add_argument("--json", default="")
    roll.set_defaults(func=run_roll)

    args = parser.parse_args()
    validate_args(parser, args)
    if getattr(args, "delay", 0) < 0:
        parser.error("--delay 不能为负数")

    global _SSL_CTX, _RUN_CACHE_KEY, _RUN_CACHE_BASE
    if args.insecure:
        _SSL_CTX = ssl._create_unverified_context()
    _RUN_CACHE_BASE = new_cache_base()
    _RUN_CACHE_KEY = args.prompt_cache_key or _RUN_CACHE_BASE
    args.func(args)


if __name__ == "__main__":
    main()
