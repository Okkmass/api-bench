#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
messages_bench.py — /v1/messages (Claude 原生格式) 端点测试工具。

同目录下的 responses_bench.py 是它的姊妹脚本，测 OpenAI 的 /v1/responses 协议。

纯标准库，无需 pip。四种模式：
  report: 一页自包含 HTML 报告 = 并发首字压测(TTFT) + 缓存持久性 + 滚动缓存。最常用。
  bench : 并发/限速压测。指标对齐 GuideLLM / LLMPerf：
          TTFT(首字) / TPOT(每输出 token) / ITL(token 间) / 吞吐(req/s, tok/s)，
          支持 p50/p90/p99、固定 RPS(Poisson 到达)、CSV 导出
  cache : 同一大前缀连发 N 次，看缓存命中率(cache_read)。
          --cache-placement both 还能诊断断点位置，识别「只认 messages 断点」这类上游缺陷
  roll  : 滚动缓存。多轮真实对话、历史逐轮累积，看命中区会不会跟着长大
          （Incremental Cache Coverage）。cache 问"缓存能不能用"，roll 问"长会话烧不烧钱"

用法:
  # 一页 HTML 报告（首字 + 缓存），并发 100
  python3 messages_bench.py --url https://你的网关/v1/messages --token sk-xxxx --model glm-5.2 \
      report --concurrency 100 --html report.html

  # 压测：并发 10，共 50 请求，导出 CSV
  python3 messages_bench.py \
      --url https://你的网关/v1/messages --token sk-xxxx --model glm-5.2 \
      bench --concurrency 10 --requests 50 --csv result.csv

  # 压测：按固定速率 5 req/s 压 100 个（Poisson 到达）
  python3 messages_bench.py --url ... --token ... --model glm-5.2 \
      bench --request-rate 5 --requests 100

  # 缓存：同一前缀发 5 次
  python3 messages_bench.py --url ... --token ... --model glm-5.2 \
      cache --rounds 5 --prefix-tokens 8000

提示:
  - 管理员固定渠道：token 末尾加 -<渠道ID>，例如 sk-xxxx-43
  - 指标口径：
      TTFT = 请求发出 → 第一个输出 token
      TPOT = (总耗时 - TTFT) / (输出 token 数 - 1)   平均每个后续 token 的耗时
      ITL  = 相邻流式 chunk 之间的间隔（逐 token 延迟分布）
      缓存命中率 = 排除首发冷启动后 cache_read>0 的请求占比
      token 命中率 = Σcache_read / Σ(input + cache_read + cache_creation)
"""

import argparse
import csv
import datetime
import html as html_lib
import http.client
import json
import math
import os
import random
import ssl
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# 可选：--insecure 时置为不校验证书的 SSL context（复用到自签名/内网端点）
_SSL_CTX = None

# ---- 请求画像与进度文件 ----
# 请求画像：claude-code=原脚本行为（伪装 Claude Code）；generic=通用 Messages 兼容接口，
# 不带 Claude Code 专用头/system 块/thinking/effort。
_PROFILE = "claude-code"

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


def check_url(parser, url):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        parser.error("--url 必须是完整的 http/https URL")
    if u.username or u.password:
        parser.error("--url 不能包含用户名或密码")

# 稳定的 Claude Code 风格 metadata.user_id：网关对它做一致性哈希→把同一 user_id 固定
# 路由到同一渠道（缓存亲和）。不带它就走随机分流、缓存被打散。
_RUN_USER_ID = None
# 固定的 base user_ 段（整个 run 内不变）。网关按 user_ 前缀（"_account_" 之前）统计"唯一用户数"，
# 同一账号窗口内不同 base 超上限（默认 50）就触发限额、连续超限会全局封禁非 root 用户。
# 压测若每请求换新 base，就会被当成上百个不同的人而触发封禁。
# 所以固定 base、只变 account/session：缓存亲和哈希的是完整 user_id 仍能把负载打散到各渠道，
# 但按 base 统计唯一用户时始终只算 1 个人。
_RUN_BASE_USER = None


def _new_base_user():
    return "user_" + uuid.uuid4().hex + uuid.uuid4().hex


def _claude_user_id(session=None):
    """构造 Claude Code 风格 user_id：<base>_account_<uuid>_session_<uuid>。

    base 段用整个 run 固定的 _RUN_BASE_USER（同一个"人"），只变 account/session：
    完整 user_id 每次不同 → 缓存亲和仍把负载打散到各渠道；base 不变 → 不触发唯一用户限额/封禁。
    """
    if _PROFILE == "generic":
        return "bench_" + uuid.uuid4().hex
    base = _RUN_BASE_USER or _new_base_user()
    sess = session or str(uuid.uuid4())
    return "%s_account_%s_session_%s" % (base, uuid.uuid4(), sess)


# 真实 Claude Code 的身份特征。取值来自抓包实测（cc_version=2.1.220.04c，见桌面
# PolarDB-X_qwen3.8-max_原始请求响应_20260804/Claude请求详解_头与体.md 第 ③④ 节和 2.3 节）。
# 严格网关不只看 6 个 x-stainless/UA 头，还会查 anthropic-beta 里有没有 claude-code-*
# 以及 system 头两块是不是 CLI 的计费上报 + 身份声明；缺了就 403
# "this endpoint only accepts requests from the official Claude Code CLI"。
_CC_VERSION = "2.1.247.cfd"

# 2026-08-27 抓包（claude-cli/2.1.247，见 ~/Desktop/claude抓包/）实测真实客户端只发 4~5 项：
#   claude-code / interleaved-thinking / mid-conversation-system / effort [/ fallback-credit]
# 这里保留更长的列表：多数是纯能力声明，"开了 beta 不填字段"不报错也没副作用，
# 多声明几项能顺带探出上游对未知 beta 的容忍度。
# 但 claude-code-20250219 是"我是 Claude Code"的声明，门卫认的就是它，不能删。
_CC_BETAS = ",".join([
    "claude-code-20250219",
    "interleaved-thinking-2025-05-14",
    "redact-thinking-2026-02-12",
    "thinking-token-count-2026-05-13",
    "context-management-2025-06-27",
    "prompt-caching-scope-2026-01-05",
    "mid-conversation-system-2026-04-07",
    "advisor-tool-2026-03-01",
    "effort-2025-11-24",
    "fallback-credit-2026-06-01",
])

# system[0]：一个伪装成 system 块的 HTTP 头（计费归因），真实 CLI 不给它打 cache_control
_CC_BILLING = ("x-anthropic-billing-header: cc_version=%s; cc_entrypoint=cli;"
               % _CC_VERSION)
# system[1]：身份声明，真实 CLI 在这里打第 1 个缓存断点
_CC_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def cc_system_preamble():
    """真实 Claude Code 的 system 前两块。恒定不变，所以不会打断后面的缓存前缀。

    不打 cache_control：断点留给各模式自己的大块（Anthropic 最多 4 个断点，省着用）。
    """
    if _PROFILE == "generic":
        return []
    return [{"type": "text", "text": _CC_BILLING},
            {"type": "text", "text": _CC_IDENTITY}]


_PARAGRAPH = (
    "You are a senior software engineer working on a large Go codebase. "
    "Follow the existing conventions, use guard clauses, handle every error, "
    "and keep changes minimal and well-scoped. Below is reference material that "
    "stays constant across the whole session so it can be cached upstream. "
)


def build_prefix(approx_tokens: int) -> str:
    """按 ~4 字符/token 估算，拼出目标大小的稳定文本。"""
    target_chars = approx_tokens * 4
    reps = max(1, target_chars // len(_PARAGRAPH) + 1)
    return (_PARAGRAPH * reps)[:target_chars]


def system_block(text: str, cache: bool):
    """CLI 身份前缀 + 压测大块。前缀恒定，排在前面也不影响后面的缓存匹配。"""
    block = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return cc_system_preamble() + [block]


def apply_cc_fields(payload: dict, args) -> dict:
    """补齐真实 Claude Code 会发、而压测脚本原本不发的 body 字段。

    ⭐ 为什么重要（2026-08-27 实测教训）：
    脚本原本只伪装了 Claude Code 的「外壳」（UA / x-stainless-* / anthropic-beta /
    system 计费块），body 里却不发 thinking 和 output_config。结果是
    **压测报告说"该渠道可用"，用户一上真 Claude Code 就 400** —— 当天所有的 400
    都是 body 内容触发的，没有一个是请求头触发的。

    三个字段各自的坑（抓包见 ~/Desktop/claude抓包/三份抓包的区别.md）：

    1) output_config.effort —— 真实 CC 默认发 "xhigh"，而智谱全线（官方
       open.bigmodel.cn + 腾讯 tokenhub）只认 low/high/max，发 xhigh 直接
       `[1210][该模型始终思考，不支持关闭思考；请使用 low、high 或 max。]`。
       注意这个报错文案完全误导：它说的是"关闭思考"，实际跟 thinking 无关。

    2) thinking.type —— 跨上游兼容性最差的字段，而且各家踩坑方向相反：
         老兼容层      拒 "adaptive"，只认 enabled/disabled
         Kimi/Moonshot 反过来拒 "disabled"，报 only type=enabled is allowed
         智谱 / 真 Claude  都接受
       真实 CC 发的是 "adaptive"，所以默认值取 adaptive 才测得准。

    3) messages 里的 role:"system"（mid-conversation system）—— 2026-04 新 beta，
       用来传"会话中途才变化的上下文"。标准协议 role 是枚举（只允许 user/assistant），
       老实现看到 system 直接 400。脚本原本在 anthropic-beta 里声明了这项能力却从不
       真的发，等于没测。

    thinking 预算和可见正文共用 max_tokens，所以 budget >= max_tokens 时自动抬高
    max_tokens，否则思考吃满预算、一个正文字都不吐 → 被记成「空响应」假失败。
    """
    # generic 画像：没有 CLI 前缀块时 system 可能是空列表，部分接口会拒，直接去掉
    if _PROFILE == "generic" and not payload.get("system"):
        payload.pop("system", None)

    # ---- thinking ----
    spec = (getattr(args, "thinking", None) or "adaptive").strip().lower()
    # 兼容旧参数：--thinking-budget N 等价于 --thinking enabled:N
    budget = getattr(args, "thinking_budget", 0) or 0
    if budget > 0:
        spec = "enabled:%d" % budget

    if spec in ("none", "off", ""):
        pass
    elif spec == "adaptive":
        payload["thinking"] = {"type": "adaptive"}
    elif spec == "disabled":
        payload["thinking"] = {"type": "disabled"}
    elif spec.startswith("enabled"):
        n = 0
        if ":" in spec:
            try:
                n = int(spec.split(":", 1)[1])
            except ValueError:
                n = 0
        payload["thinking"] = {"type": "enabled"} if n <= 0 else \
            {"type": "enabled", "budget_tokens": n}
        if n > 0 and payload.get("max_tokens", 0) <= n:
            payload["max_tokens"] = n + 256
    else:
        payload["thinking"] = {"type": spec}      # 直接透传，方便试非标值

    # ---- output_config.effort ----
    effort = (getattr(args, "effort", None) or "xhigh").strip().lower()
    if effort not in ("none", "off", ""):
        payload["output_config"] = {"effort": effort}

    # ---- mid-conversation system ----
    if getattr(args, "mid_conv_system", False):
        msgs = payload.get("messages")
        if isinstance(msgs, list) and msgs:
            msgs.append({"role": "system",
                         "content": "Available agent types for the Agent tool:\n"
                                    "- general-purpose: 通用代理。\n"
                                    "(mid-conversation system 探针，用于验证上游是否接受 "
                                    "messages 里的 role:\"system\")"})
    return payload


def cache_messages(bulk: str, tail: str, placement: str = "messages"):
    """构造缓存测试 messages。placement 决定稳定大块放哪。

    ⚠️ 历史结论已被推翻，别再当通用规律用：
    本函数原先写死「大块**必须**放 messages，放 system 恒 0% 命中」，依据是
    2026-08-04 PolarDB-X AI 网关 / qwen3.8-max 的实测——那边大块放 system 时每发
    都记一笔 cache_creation 却从不读回，连逐字节相同的请求也是 read=0。
    但 2026-08-27 用同一份 payload 复测真 Claude（claude-opus-5）和智谱官方
    （open.bigmodel.cn/api/anthropic，glm-5.3-flash），**放 system 一样正常命中**
    （抓包见 ~/Desktop/claude抓包/）。

    所以那是 PolarDB-X 那个网关自己的实现缺陷，不是协议规律。
    保留 messages 为默认值（最保守、各家都认），但把 system 放法开放出来 ——
    用 --cache-placement both 跑一轮，就能直接诊断出「这个上游只认 messages 断点」
    这类缺陷，这比闷头用保守写法更有信息量。

    placement:
      messages —— 大块 = messages[0] 且断点打在它上面（默认，兼容性最好）
      system   —— 大块进 system 并在那里打断点，messages 只留变化的尾巴
      注：both 由调用方交替传 messages/system 实现，本函数只认单个值。
    """
    if placement == "system":
        # 大块不在 messages 里；调用方需自行把 system 换成 system_block(bulk, cache=True)
        return [{"role": "user", "content": [{"type": "text", "text": tail}]}]
    return [
        {"role": "user", "content": [{"type": "text", "text": bulk,
                                      "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "收到，请继续。"}]},
        {"role": "user", "content": [{"type": "text", "text": tail}]},
    ]


# 每线程复用 keep-alive 连接：避免每个请求都做一次 TCP+TLS 握手把握手时间混进 TTFT
_TLS = threading.local()


def _drop_conn():
    conn = getattr(_TLS, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa
            pass
    _TLS.conn = None
    _TLS.key = None


def _get_conn(scheme, host, port, timeout):
    """返回 (连接, 是否复用)。同线程同端点复用；--insecure 时用不校验的 SSL context。"""
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


def send_stream(url: str, token: str, payload: dict, timeout: int = 180) -> dict:
    """发一个流式请求，返回 TTFT / 总耗时 / ITL 列表 / usage / 文本。

    连接按线程复用（keep-alive）；timeout 既是 socket 超时也是整条流的总 deadline。
    """
    data = json.dumps(payload).encode("utf-8")
    u = urlparse(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": token,                    # Claude 原生鉴权
        "Authorization": "Bearer " + token,    # 兼容 Bearer
        "anthropic-version": "2023-06-01",
        "accept": "text/event-stream",          # 本脚本恒流式；严格网关会按它判路由
        "connection": "keep-alive",             # 对齐抓包；连接本来就复用，带上更像真实 CLI
        # 声明要开的能力。对齐真实 Claude Code 的完整 9 项：严格网关会查里面有没有
        # claude-code-20250219，只带 prompt-caching-* 会被判成非官方 CLI 并 403
        "anthropic-beta": _CC_BETAS,
        # 伪装 Claude CLI，避免被渠道 UA / x-stainless 校验拦截。
        # 取值对齐真实 Claude Code：os=darwin、package-version 为合法 semver（假值如 0.0.0 会被严格网关判伪）
        "user-agent": "claude-cli/1.0.5 (external, cli)",
        "x-stainless-lang": "js",
        "x-stainless-os": "darwin",
        "x-stainless-runtime": "node",
        "x-stainless-runtime-version": "v20.0.0",
        "x-stainless-package-version": "0.26.1",
    }

    if _PROFILE == "generic":
        # 通用画像：只保留协议必需头，去掉 Claude Code 的 beta/UA/stainless 伪装
        for k in ("anthropic-beta", "user-agent", "x-stainless-lang", "x-stainless-os",
                  "x-stainless-runtime", "x-stainless-runtime-version",
                  "x-stainless-package-version"):
            headers.pop(k, None)
        headers["user-agent"] = "api-bench/1.0"

    start = time.time()
    resp = None
    for attempt in (0, 1):
        conn, reused = _get_conn(u.scheme, u.hostname, port, timeout)
        try:
            conn.request("POST", path, body=data, headers=headers)
            resp = conn.getresponse()
            break
        except (http.client.HTTPException, OSError) as e:
            _drop_conn()
            # 复用的连接可能已被服务端空闲断开，换新连接重试一次
            if reused and attempt == 0:
                continue
            return {"error": "%s: %s" % (type(e).__name__, e)}
    if resp.status != 200:
        try:
            body = resp.read().decode(errors="replace")
        except Exception:  # noqa
            body = ""
            _drop_conn()
        return {"error": "HTTP %d: %s" % (resp.status, body[:400])}

    ttft = None
    last_t = None
    itls = []
    usage = {}
    text_parts = []
    broke_early = False
    stop_reason = None   # 空响应时唯一能定性的证据，见函数末尾
    try:
        while True:
            raw = resp.readline()
            if not raw:
                break
            # 总 deadline：socket 级 timeout 只管单次读，防不住"每秒滴一个字"的僵尸流
            if time.time() - start > timeout:
                _drop_conn()
                return {"error": "整体超时 %ds（流未结束）" % timeout}
            line = raw.decode(errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                broke_early = True
                break
            try:
                obj = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "error":
                # HTTP 200 里夹带的流内错误事件（如 overloaded_error）：必须记为错误，
                # 否则没有 error 字段、会被 bench/report 当成"成功请求"，压低错误率、抬高吞吐
                _drop_conn()
                eo = obj.get("error", {}) or {}
                return {"error": "流内错误 %s: %s"
                        % (eo.get("type", "error"), str(eo.get("message", ""))[:200])}
            if t == "message_start":
                usage.update(obj.get("message", {}).get("usage", {}) or {})
            elif t == "content_block_delta":
                now = time.time()
                if ttft is None:
                    ttft = now - start
                elif last_t is not None:
                    itls.append(now - last_t)
                last_t = now
                delta = obj.get("delta", {})
                if "text" in delta:
                    text_parts.append(delta["text"])
            elif t == "message_delta":
                usage.update(obj.get("usage", {}) or {})
                sr = (obj.get("delta", {}) or {}).get("stop_reason")
                if sr:
                    stop_reason = sr
    except (http.client.HTTPException, OSError) as e:
        _drop_conn()
        return {"error": "流中断 %s: %s" % (type(e).__name__, e)}
    if broke_early:
        _drop_conn()  # 没读到 EOF，连接上有残余数据，不能复用
    else:
        resp.close()

    total = time.time() - start
    if ttft is None:
        # HTTP 200 但整条流没有任何输出内容 → 记为错误，别混进成功统计。
        #
        # 但「无输出」有两种性质截然不同的原因，报错必须把证据带出来，否则排查会跑偏：
        #   A. 预算耗尽（配置问题）：推理模型先花预算思考，部分网关把推理 token 计进
        #      output_tokens 却不下发任何 delta，预算在思考结束前用完 → stop_reason=max_tokens
        #      且 output_tokens 恰好等于 max_tokens。调大 --max-tokens 即可，**降并发无效**。
        #   B. 上游静默失败（协议问题）：真的什么都没产出。
        # 实测教训（2026-08-26）：旧版只报「空响应」三个字，害得人去跑并发梯度找容量拐点，
        # 而失败率根本不随并发变化（10/30/60 并发都约 50%）——白折腾一轮。
        out_tok = usage.get("output_tokens") or 0
        if stop_reason == "max_tokens":
            return {"error": "预算耗尽（思考吃满 max_tokens，output_tokens=%d，无正文，%.1fs）"
                             "→ 调大 --max-tokens，这是配置问题，降并发无效" % (out_tok, total),
                    "err_class": "budget_exhausted"}
        return {"error": "空响应（HTTP 200 无输出，stop_reason=%s，output_tokens=%d，%.1fs）"
                         % (stop_reason or "无", out_tok, total),
                "err_class": "empty_response"}
    out_tokens = usage.get("output_tokens") or 0
    tpot = None
    if out_tokens > 1:
        tpot = (total - ttft) / (out_tokens - 1)
    return {
        "ttft": ttft,
        "total": total,
        "tpot": tpot,
        "itls": itls,
        "usage": usage,
        "text": "".join(text_parts),
    }


# 失败分类表：(类别, 判定关键字, 性质)。性质决定了下一步该往哪查——
# 把配置问题误读成容量问题，就会去降并发、查限流，而那两件事一点用都没有。
_ERR_CLASSES = [
    ("budget_exhausted", "预算耗尽",   "配置问题"),
    ("empty_response",   "空响应",     "协议问题"),
    ("timeout",          "整体超时",   "容量问题"),
    ("stream_broken",    "流中断",     "容量问题"),
    ("api_error",        "流内错误",   "协议问题"),
]


def classify_error(msg: str):
    """把错误原文归到一个类别，返回 (类别, 性质)。"""
    m = str(msg)
    for name, keyword, nature in _ERR_CLASSES:
        if keyword in m:
            return name, nature
    if m.startswith("HTTP 429"):
        return "http_429", "容量问题"
    if m.startswith("HTTP 5"):
        return "http_5xx", "容量问题"
    if m.startswith("HTTP 4"):
        return "http_4xx", "配置问题"
    return "conn_failed", "容量问题"


def print_error_buckets(errors):
    """按类别打印失败分布，并指出本轮结果能/不能证明什么。"""
    if not errors:
        return
    buckets = {}
    for e in errors:
        name, nature = classify_error(e)
        buckets.setdefault((name, nature), 0)
        buckets[(name, nature)] += 1
    print("\n----- 失败分类 -----")
    for (name, nature), n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print("  %-18s x%-4d [%s]" % (name, n, nature))
    if any(name == "budget_exhausted" for name, _ in buckets):
        print("  → budget_exhausted 是配置问题：调大 --max-tokens，降并发不会让它变好。")
    if not any(nature == "容量问题" for _, nature in buckets):
        print("  → 没有任何容量类失败，本轮结果不能作为并发上限的证据。")


def preflight(args, send_one, label="预检"):
    """开压前先单发一次，失败就带原因中止。

    不做预检的代价：鉴权失败、模型名写错、余额不足这类问题会以 N 条一模一样的错误
    刷满整轮压测（实测 100 并发跑满才发现是「apikey 可用额度不足」），既浪费时间也
    浪费额度。单发一次几秒钟，能把这类问题挡在门外。
    """
    print("[%s] 单发一次确认端点可用 ..." % label)
    print("[%s] 发送 thinking=%s  effort=%s  mid_conv_system=%s"
          % (label, getattr(args, "thinking", "adaptive") or "adaptive",
             getattr(args, "effort", "xhigh") or "xhigh",
             "是" if getattr(args, "mid_conv_system", False) else "否"))
    r = send_one()
    # effort=xhigh 被 400 拒绝时（如 sonnet-4-6 只支持 low/medium/high/max），降到 high 重试一次
    if "HTTP 400" in str(r.get("error", "")) and (getattr(args, "effort", "") or "").lower() == "xhigh":
        print("[%s] effort=xhigh 被拒绝，改用 high 重试" % label)
        args.effort = "high"
        r = send_one()
    if "error" in r:
        print("[%s] 失败，已中止（不再发起并发压测）：\n  %s" % (label, r["error"]))
        _preflight_hint(args, r["error"])
        return False
    u = r.get("usage", {}) or {}
    print("[%s] 通过：首字 %.0fms，%s"
          % (label, (r.get("ttft") or 0) * 1000, fmt_usage(u)))
    print("[%s] 上游成色：%s" % (label, upstream_grade(u)))
    return True


def _preflight_hint(args, err: str):
    """把已知的兼容性 400 翻译成人话 + 给出下一步。

    这些都是 2026-08-27 实测踩过的坑，报错文案本身普遍误导，
    不给提示的话很容易往错误方向排查（我第一次撞 1210 就往 thinking 查了半天）。
    """
    e = str(err)
    effort = (getattr(args, "effort", None) or "xhigh").lower()
    if "1210" in e or "不支持关闭思考" in e or "low、high 或 max" in e:
        print("  ↳ 提示：这个报错文案是误导的，跟 thinking 无关。")
        if effort == "xhigh":
            print("     真实原因通常是 output_config.effort=\"xhigh\" 不在上游白名单"
                  "（智谱全线只认 low/high/max）。")
            print("     重试：加 --effort high")
        else:
            print("     当前 effort=%s 已不是 xhigh，若仍报此错，试 --thinking enabled:2048"
                  % effort)
    elif "thinking" in e.lower():
        print("  ↳ 提示：thinking 字段不被接受。老实现只认 enabled/disabled，")
        print("     Kimi/Moonshot 系则反过来拒 disabled。重试：--thinking enabled:2048"
              " 或 --thinking none")
    elif "role" in e.lower() and getattr(args, "mid_conv_system", False):
        print("  ↳ 提示：上游不接受 messages 里的 role:\"system\"（老 schema 把 role 当枚举）。")
        print("     去掉 --mid-conv-system 再试。")


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def stat_line(label, values, unit="ms", scale=1000.0):
    if not values:
        print("%-12s (无数据)" % label)
        return
    print("%-12s p50=%.1f%s  p90=%.1f%s  p99=%.1f%s  mean=%.1f  min=%.1f  max=%.1f"
          % (label,
             pct(values, 50) * scale, unit,
             pct(values, 90) * scale, unit,
             pct(values, 99) * scale, unit,
             statistics.mean(values) * scale,
             min(values) * scale, max(values) * scale))


def fmt_usage(u: dict) -> str:
    s = "input=%s  cache_read=%s  cache_creation=%s  output=%s" % (
        u.get("input_tokens", "-"),
        u.get("cache_read_input_tokens", "-"),
        u.get("cache_creation_input_tokens", "-"),
        u.get("output_tokens", "-"),
    )
    th = (u.get("output_tokens_details") or {}).get("thinking_tokens")
    if th is not None:
        s += "  thinking=%s" % th
    return s


def upstream_grade(u: dict) -> str:
    """按 usage 字段的丰富度给上游"成色"打标。

    依据 2026-08-27 三方抓包（~/Desktop/claude抓包/三份抓包的区别.md）：
    只有真 Claude 会返回 thinking_tokens 和 cache_creation 的 5m/1h 分档；
    智谱官方有 signature_delta / service_tier 但没有这两个；
    浅层兼容实现则只给 input_tokens / output_tokens。

    注意：signature_delta 不能用来判断"是不是真 Claude"——智谱官方也有。
    """
    marks = []
    if (u.get("output_tokens_details") or {}).get("thinking_tokens") is not None:
        marks.append("thinking_tokens")
    if isinstance(u.get("cache_creation"), dict):
        marks.append("缓存5m/1h分档")
    if u.get("service_tier"):
        marks.append("service_tier")
    if u.get("inference_geo"):
        marks.append("inference_geo")
    if u.get("cache_read_input_tokens") is not None:
        marks.append("缓存字段")
    if not marks:
        return "浅层兼容（只有 input/output，连缓存字段都不返回）"
    return "、".join(marks)


def run_bench(args):
    prefix = build_prefix(args.prefix_tokens)
    rate = float("inf") if args.request_rate == "inf" else float(args.request_rate)
    print("[bench] model=%s concurrency=%d requests=%d rate=%s prefix~%d tok"
          % (args.model, args.concurrency, args.requests, args.request_rate, args.prefix_tokens))

    def one(i):
        payload = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "stream": True,
            "system": system_block(prefix, cache=True),
            "messages": [{"role": "user", "content": "用一句话回答：这是第 %d 个请求。" % i}],
            "metadata": {"user_id": _RUN_USER_ID},
        }
        return send_stream(args.url, args.token, apply_cc_fields(payload, args), args.timeout)

    # 并发压测前先探一次；单发模式（requests<=1）本身就是探测，不必多花一次
    _pre = args.requests > 1 and not args.no_preflight
    progress_plan((["preflight"] if _pre else []) + ["load"])
    if _pre:
        progress("preflight", 0, 1)
        if not preflight(args, lambda: one(-1)):
            return
        progress("preflight", 1, 1)

    results = {}
    wall_start = time.time()
    progress("load", 0, args.requests)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut_to_idx = {}
        for i in range(args.requests):
            fut_to_idx[ex.submit(one, i)] = i
            if rate != float("inf") and i < args.requests - 1:
                # Poisson 到达：指数分布的到达间隔
                time.sleep(random.expovariate(rate))
        for fut in as_completed(fut_to_idx):
            results[fut_to_idx[fut]] = fut.result()
            progress("load", len(results), args.requests)
    wall = time.time() - wall_start

    ttfts, totals, tpots, all_itls, errors = [], [], [], [], []
    total_output = 0
    rows = []
    for i in range(args.requests):
        r = results.get(i, {"error": "missing"})
        if "error" in r:
            errors.append(r["error"])
            rows.append({"idx": i, "status": "error", "ttft_ms": "", "total_ms": "",
                         "tpot_ms": "", "mean_itl_ms": "", "input": "", "cache_read": "",
                         "output": "", "error": r["error"]})
            continue
        u = r["usage"]
        out = u.get("output_tokens") or 0
        total_output += out
        if r["ttft"] is not None:
            ttfts.append(r["ttft"])
        totals.append(r["total"])
        if r["tpot"] is not None:
            tpots.append(r["tpot"])
        all_itls.extend(r["itls"])
        mean_itl = statistics.mean(r["itls"]) if r["itls"] else None
        rows.append({
            "idx": i, "status": "ok",
            "ttft_ms": round(r["ttft"] * 1000, 1) if r["ttft"] is not None else "",
            "total_ms": round(r["total"] * 1000, 1),
            "tpot_ms": round(r["tpot"] * 1000, 1) if r["tpot"] is not None else "",
            "mean_itl_ms": round(mean_itl * 1000, 1) if mean_itl is not None else "",
            "input": u.get("input_tokens", ""),
            "cache_read": u.get("cache_read_input_tokens", ""),
            "output": out,
            "error": "",
        })

    ok = len(totals)
    print("\n===== 压测结果 =====")
    print("成功: %d   失败: %d   总墙钟: %.2fs" % (ok, len(errors), wall))
    print("请求吞吐: %.2f req/s   输出吞吐: %.1f tok/s   总输出: %d tok"
          % (ok / wall if wall else 0, total_output / wall if wall else 0, total_output))
    print()
    stat_line("TTFT(首字)", ttfts)
    stat_line("TPOT/token", tpots)
    stat_line("ITL(token间)", all_itls)
    stat_line("总耗时", totals)
    print_error_buckets(errors)
    if errors:
        print("\n前几个错误样本:")
        for e in errors[:5]:
            print("  -", e)

    if getattr(args, "json", ""):
        # 与 report --json 同结构，bench 模式没有 cache/roll 段
        meta = {"model": args.model, "url": display_url(args.url), "concurrency": args.concurrency,
                "max_tokens": args.max_tokens, "prefix_tokens": args.prefix_tokens,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _b = {"ok": ok, "fail": len(errors), "wall": wall,
              "req_throughput": ok / wall if wall else 0,
              "out_throughput": total_output / wall if wall else 0,
              "tps_mean": (1.0 / statistics.median(tpots)) if tpots else 0,
              "ttft": ttfts, "tpot": tpots, "total": totals}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report_json(meta, _b, None, None, errors), f, ensure_ascii=False, indent=2)
        print("JSON 已写入: %s" % args.json)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "status", "ttft_ms", "tpot_ms",
                                              "mean_itl_ms", "total_ms", "input",
                                              "cache_read", "output", "error"])
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print("\nCSV 已写入: %s" % args.csv)


def fake_tools(n: int):
    """伪造 n 个工具定义，用来把请求体撑到真实 Claude Code 的形态。

    为什么必须带：2026-08-27 三份抓包实测，tools 占请求体 83~86%
    （真 Claude 106 KB / 54 个，智谱官方 115 KB / 58 个）。
    不带 tools 的滚动测试，"省了多少钱"这个结论是虚的 —— 真实场景里最大的一块
    恰恰是 tools，它能不能进缓存直接决定成本模型。
    """
    out = []
    for i in range(n):
        out.append({
            "name": "probe_tool_%02d" % i,
            "description": ("Reference tool #%d used only to pad the request to a realistic "
                            "Claude Code shape. It documents conventions, error handling rules, "
                            "and edge cases so the definition is long enough to matter. " % i) * 3,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to operate on."},
                    "pattern": {"type": "string", "description": "Optional filter pattern."},
                    "limit": {"type": "integer", "description": "Maximum rows to return."},
                },
                "required": ["path"],
            },
        })
    return out


def run_roll(args):
    """滚动缓存测试：多轮对话，看命中区是否随对话变长而增长。

    和 cache 模式的区别 —— 这是两个不同的问题：
      cache 模式  固定大前缀反复发，问"缓存能不能用"
      roll  模式  历史一轮轮追加，问"缓存能不能跟着对话长大"
    后者才决定长会话的真实成本。固定前缀测法很容易跑出 100% 命中率，
    但那只说明第一个断点有效，中间增长的历史可能一点没进缓存。

    ⭐ 断点必须打在【已固定的历史末尾】，不是新问题上。
    前缀缓存匹配的是"开头到断点"这一段，断点落在每轮都变的内容上等于白打 ——
    我第一版就是这么写的，结果 cache_read 横盘在 1500 不动，误以为"上游不支持增量"。
    改成打在上一轮的 assistant 回复上，增长立刻出来了。

        msgs = [大块(断点1)] [历史…] [上轮回复(断点2)] [本轮新问题]
                                          ↑ 每轮往后移，把新固定下来的历史圈进缓存

    三种失败态要分开判（2026-08-27 三份抓包的真实形态）：
      cache_read 恒 0        → 上游根本不支持缓存（onelinkai 连字段都不返回）
      cache_read 恒定不变    → 只有固定断点生效，历史没进缓存 ← 最阴险，命中率看着很好看
      cache_read 逐轮变大    → 正常（真 Claude 46,328→122,317；智谱 1,472→1,600）

    中间偶发单点 miss 属正常（多副本路由落到没有该缓存的副本），
    真 Claude 第 23 发、智谱第 3 发都出现过，所以看"递增轮次占比"而不是"有没有 miss"。
    """
    rows = roll_collect(args, verbose=True)
    if rows:
        roll_summary(rows, args)


def roll_prepare(args, verbose=True, turns=None, tag="roll"):
    """滚动测试的准备：拼好请求构造器。不发请求，所以可以提前做。"""
    turns = turns or args.turns
    n_tools = getattr(args, "tools", 20)
    turn_tokens = getattr(args, "turn_tokens", 0)
    tools = fake_tools(n_tools) if n_tools > 0 else None
    bulk = ("[%s-%d]\n\n" % (tag, os.getpid())) + build_prefix(args.prefix_tokens)
    filler = (" " + build_prefix(turn_tokens)) if turn_tokens > 0 else ""

    tool_bytes = len(json.dumps(tools, ensure_ascii=False)) if tools else 0
    if verbose:
        print("[roll] model=%s turns=%d tools=%d(%.0f KB) 大块~%d tok 每轮填充~%d tok"
              % (args.model, turns, n_tools, tool_bytes / 1024.0,
                 args.prefix_tokens, turn_tokens))
        print("断点布局: system[1] · system[2] · messages滚动 —— 3 个，和真实 Claude Code 一致")
        print("说明: 滚动断点打在已固定的历史末尾。"
              "健康表现 = cache_read 逐轮变大、付费 input 保持很小。\n")

    # ⭐ 断点布局必须复刻真实 Claude Code 的 3 个，位置也要一样：
    #     tools(不打)  system[0]计费头(不打)  system[1]【断点1】
    #     system[2] 长规则【断点2】
    #     messages[已固定末尾]【断点3，每轮往后滚】
    #
    # 早期版本把大块塞在 messages[0] 上打断点1、滚动断点当断点2，只有 2 个断点。
    # 那样测不出"上游认不认第 3 个断点"——而真实客户端用的正是第 3 个。
    # 实测 tokenhub：2 断点布局下滚动缓存 109% 健康，换成真实的 3 断点布局后，
    # 真实 Claude Code 会话里 cache_read 卡死 14 轮不动。测出来的是两回事。
    system_blocks = [
        {"type": "text", "text": _CC_BILLING},                       # system[0] 不打
        {"type": "text", "text": _CC_IDENTITY,                       # system[1] 断点1
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": bulk,                               # system[2] 断点2
         "cache_control": {"type": "ephemeral"}},
    ]
    if _PROFILE == "generic":
        # 通用画像：保持 3 个断点（system×2 + messages 滚动），但不带 CLI 身份块
        system_blocks = [
            {"type": "text", "text": "You are a helpful assistant.",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": bulk, "cache_control": {"type": "ephemeral"}},
        ]
    history = [
        {"role": "user", "content": [{"type": "text", "text": "你好，我们开始。"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "收到，请继续。"}]},
    ]

    def build(turn):
        msgs = json.loads(json.dumps(history))          # 深拷贝，别污染 history
        # 断点3：滚动打在已固定的历史最后一条上
        last = msgs[-1]["content"]
        if isinstance(last, list) and last:
            last[-1]["cache_control"] = {"type": "ephemeral"}
        q = "第%d轮：用一句话回答，%d 加 %d 等于几。%s" % (turn, turn, turn, filler)
        msgs.append({"role": "user", "content": [{"type": "text", "text": q}]})
        payload = {
            "model": args.model, "max_tokens": args.max_tokens, "stream": True,
            "system": system_blocks,
            "messages": msgs,
            "metadata": {"user_id": _RUN_USER_ID},
        }
        if tools:
            payload["tools"] = tools
        return payload, q

    return {"turns": turns, "build": build, "history": history, "warm_thread": None}


ROLL_WARM_LEAD_S = 30


def roll_warm(args, ctx, verbose=True, delay=None):
    """预热：写入对后续请求可见有延迟（实测 DeepSeek 第1发就命中，真 Claude 第4发，智谱第7发）"""
    warm_n = getattr(args, "warmup", 2)
    if delay is None:
        delay = getattr(args, "delay", 1.5)
    if warm_n > 0:
        if verbose:
            print("预热 %d 次..." % warm_n)
        for _ in range(warm_n):
            send_stream(args.url, args.token, apply_cc_fields(ctx["build"](0)[0], args), args.timeout)
            if delay:
                time.sleep(delay)
        if verbose:
            print()


def roll_collect(args, verbose=True, turns=None, tag="roll", ctx=None, warm_delay=None):
    """滚动测试的核心循环，返回每轮的 usage 明细。

    抽出来是为了让 report 模式复用同一套逻辑 —— 报告里的滚动曲线和单独跑 roll
    必须是同一个口径，否则两边数字对不上就没法互相印证。

    ctx：report 可以先 roll_prepare，再把 roll_warm 放到缓存采样窗口末尾的后台线程里
    提前做（ctx["warm_thread"]），这里只需等它做完；没有提前做就在这里同步预热。
    """
    if ctx is None:
        ctx = roll_prepare(args, verbose, turns, tag)
    turns, build, history = ctx["turns"], ctx["build"], ctx["history"]
    delay = getattr(args, "delay", 1.5)
    if ctx["warm_thread"] is not None:
        ctx["warm_thread"].join()
    else:
        roll_warm(args, ctx, verbose, warm_delay)

    if verbose:
        print("%-6s %10s %10s %12s %12s %8s %10s"
              % ("轮次", "总输入", "付费input", "cache_read", "cache_creat", "命中率", "命中增量"))
    rows, prev_read = [], None
    progress("roll", 0, turns)
    for turn in range(1, turns + 1):
        payload, q = build(turn)
        r = send_stream(args.url, args.token, apply_cc_fields(payload, args), args.timeout)
        if "error" in r:
            print("第%-4d 错误: %s" % (turn, r["error"]))
            if turn == 1:
                _preflight_hint(args, r["error"])
                return []
            break
        u = r.get("usage") or {}
        inp = u.get("input_tokens") or 0
        read = u.get("cache_read_input_tokens")
        crea = u.get("cache_creation_input_tokens")
        total = inp + (read or 0) + (crea or 0)
        rate = (read / total * 100) if (read and total) else 0.0
        grow = "-" if prev_read is None else "%+d" % ((read or 0) - prev_read)
        if verbose:
            print("第%-4d %10s %10s %12s %12s %7.1f%% %10s"
                  % (turn, "{:,}".format(total), "{:,}".format(inp),
                     "{:,}".format(read) if read is not None else "(字段缺失)",
                     "{:,}".format(crea) if crea is not None else "(缺失)", rate, grow))
        rows.append({"turn": turn, "total": total, "input": inp,
                     "read": read or 0, "creation": crea or 0,
                     "has_read_field": read is not None})
        prev_read = read or 0
        progress("roll", turn, turns)
        # 把本轮问答追加进历史，下一轮它就成了稳定前缀的一部分
        history.append({"role": "user", "content": [{"type": "text", "text": q}]})
        history.append({"role": "assistant",
                        "content": [{"type": "text", "text": r.get("text") or "（空）"}]})
        time.sleep(delay)
    return rows


def roll_stats(rows):
    """把滚动明细压成一组可判定的指标（终端和 HTML 报告共用同一口径）。"""
    reads = [x["read"] for x in rows]
    totals = [x["total"] for x in rows]
    sum_total = sum(totals)
    sum_input = sum(x["input"] for x in rows)

    # ⭐ 判据是"命中区增长 ÷ 历史增长"，不是"递增轮次占比"。
    # 第一版用后者，被缓存块粒度坑了：DeepSeek 按 128 token 成块，而每轮只追加 ~25 token
    # （短问答 + max_tokens=64），要攒 5 轮才跨过一个块边界 → 序列看着"横盘"，
    # 实际 8 轮里历史涨了 178、命中区涨了 128，捕获率 72%，是健康的。
    # 冷启动那几轮也要剔掉：首轮必然 miss，算进基数会让捕获率虚高到 300%+
    # （实测 DeepSeek 首轮 read=128、次轮 6,272，差值全被算成"增长"）。
    warm = next((i for i, x in enumerate(rows) if x["read"] >= x["total"] * 0.5), None)
    if warm is None or warm >= len(rows) - 1:
        warm = 0
    hist_growth = totals[-1] - totals[warm]
    read_growth = reads[-1] - reads[warm]
    capture = (read_growth / hist_growth) if hist_growth > 0 else 0.0

    # 样本太少时不下结论。n=2 且首轮是冷启动的话，hist_growth 只有几百、
    # read_growth 却是整个大块（上万），捕获率算出来是 2467% 这种数，
    # 判语还会给 "✅ 滚动缓存正常" —— 明明什么都没测出来。
    if len(rows) - warm < 3:
        verdict, level = ("有效轮次只有 %d 轮（剔除冷启动后 %d 轮），样本不足，不下结论"
                          % (len(rows), len(rows) - warm)), "warn"
    elif not any(x["has_read_field"] for x in rows):
        verdict, level = "上游连 cache_read_input_tokens 字段都不返回 —— 不支持 prompt cache", "bad"
    elif max(reads) == 0:
        verdict, level = "全程 cache_read=0 —— 缓存未生效", "bad"
    elif capture >= 0.5:
        verdict, level = "命中区随对话增长（捕获 %.0f%%），滚动缓存正常" % (capture * 100), "ok"
    elif capture > 0.15:
        verdict, level = "部分增长（捕获 %.0f%%）—— 可能只是缓存块粒度导致的滞后" % (capture * 100), "warn"
    else:
        verdict, level = ("命中区几乎不增长 —— 只有固定断点生效，增长的历史没进缓存；"
                          "命中率看着好看，但长会话成本会随轮数线性上涨"), "warn"

    return {
        "rows": rows, "reads": reads, "totals": totals,
        "warm_turn": rows[warm]["turn"], "hist_growth": hist_growth,
        "read_growth": read_growth, "capture": capture,
        "per_turn": hist_growth / max(1, len(rows) - 1 - warm),
        "sum_total": sum_total, "sum_input": sum_input,
        "saved": (1 - sum_input / sum_total) * 100 if sum_total else 0,
        "verdict": verdict, "level": level,
    }


def roll_summary(rows, args):
    s = roll_stats(rows)
    reads = s["reads"]
    totals = [x["total"] for x in rows]
    sum_total, sum_input = sum(totals), sum(x["input"] for x in rows)
    saved = (1 - sum_input / sum_total) * 100 if sum_total else 0

    print("\n===== 滚动缓存结论 =====")
    print("轮数: %d   命中区: %s → %s (%+d)"
          % (len(rows), "{:,}".format(reads[0]), "{:,}".format(reads[-1]),
             s["read_growth"] if s["warm_turn"] == rows[0]["turn"] else reads[-1] - reads[0]))
    print("统计口径: 第%d轮起(剔除冷启动)   历史增长 %+d tok (每轮约 %.0f)   命中区捕获 %.0f%%"
          % (s["warm_turn"], s["hist_growth"], s["per_turn"], s["capture"] * 100))
    print("累计总输入: %s   累计付费 input: %s   省下 %.1f%%"
          % ("{:,}".format(s["sum_total"]), "{:,}".format(s["sum_input"]), s["saved"]))
    icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}[s["level"]]
    print("判定: %s %s" % (icon, s["verdict"]))

    if s["per_turn"] < 200:
        print("提示: 每轮只增长约 %.0f tok，小于常见的缓存块粒度(64~128)，结论容易被粒度噪声干扰。"
              % s["per_turn"])
        print("      想要更干脆的曲线：加 --turn-tokens 500 让每轮多塞点内容。")


def run_cache(args):
    # 每轮跑批用独立冷前缀。标记必须放最前面：前缀缓存按开头字节匹配，
    # 放中间/末尾等于没分叉。不加标记会复用上一次跑批（尤其 report ① 的高并发）
    # 焐热的缓存，命中率虚高到 100%，把"别人焐好的缓存还热不热"误报成"本网关缓存很好"。
    # 注意也不能靠改 --prefix-tokens 来换新前缀：8123 的前 32000 字符和 8000 完全相同，照样命中。
    prefix = ("[cache-test-%d]\n\n" % os.getpid()) + build_prefix(args.prefix_tokens)
    print("[cache] model=%s requests=%d concurrency=%d warmup=%d prefix~%d tok"
          % (args.model, args.rounds, args.concurrency, args.warmup, args.prefix_tokens))
    print("说明: 同一前缀连发。前 1~2 次 miss 属正常——写入对后续请求可见有一发延迟。")
    print("      并发>1 时首批会并行抢跑、都 miss；warmup 可先预热把缓存写进去。")
    print("      期间偶发 creation!=0 的重写多为多副本路由（落到没有该缓存的副本）。\n")

    def one(i, placement=None):
        place = placement or getattr(args, "cache_placement", "messages")
        payload = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "stream": True,
            # placement=system 时大块进 system 并在那里打断点；否则只放 CLI 身份前缀过门卫
            "system": (system_block(prefix, cache=True) if place == "system"
                       else cc_system_preamble()),
            # 断点放哪由 --cache-placement 决定 —— 详见 cache_messages() 的说明
            "messages": cache_messages(prefix, "缓存测试请求 #%d，简短回答即可。" % i, place),
            "metadata": {"user_id": _RUN_USER_ID},
        }
        return send_stream(args.url, args.token, apply_cc_fields(payload, args), args.timeout)

    # --cache-placement both：先做一次断点位置诊断，再按 messages 放法跑正式统计。
    # 目的是识别「只认 messages 断点」这类上游缺陷（PolarDB-X 网关就是这样）。
    if getattr(args, "cache_placement", "messages") == "both":
        print("[诊断] 对比断点放 messages / 放 system 两种写法（各发 4 次，取后 2 次）")
        verdict = {}
        for place in ("messages", "system"):
            reads = []
            for k in range(4):
                r = one(-100 - k, placement=place)
                if "error" in r:
                    print("  %-8s 第%d发 错误: %s" % (place, k + 1, r["error"]))
                    break
                reads.append((r.get("usage") or {}).get("cache_read_input_tokens") or 0)
            verdict[place] = reads
            tail = reads[2:] or reads
            print("  %-8s cache_read 序列=%s  后段命中=%s"
                  % (place, reads, "是" if any(x > 0 for x in tail) else "否"))
        m_ok = any(x > 0 for x in (verdict.get("messages") or [])[2:])
        s_ok = any(x > 0 for x in (verdict.get("system") or [])[2:])
        if m_ok and s_ok:
            print("  → 两种放法都命中，缓存实现正常\n")
        elif m_ok and not s_ok:
            print("  → ⚠️ 只认 messages 断点，放 system 不读回：该上游缓存实现有缺陷\n")
        elif s_ok and not m_ok:
            print("  → ⚠️ 只认 system 断点，放 messages 不读回（少见，请复核）\n")
        else:
            print("  → ⚠️ 两种放法都没命中，该上游可能根本不支持 prompt cache\n")

    # 预热：先打 warmup 次把缓存写进去（不计入统计）
    if args.warmup > 0:
        print("预热 %d 次..." % args.warmup)
        for i in range(args.warmup):
            one(-1 - i)

    verbose = args.rounds <= 20
    indexed = {}  # idx -> result
    if args.concurrency <= 1:
        for i in range(args.rounds):
            r = one(i)
            indexed[i] = r
            if "error" in r:
                print("  #%d  错误: %s" % (i + 1, r["error"]))
            elif verbose:
                print("  #%d  %s" % (i + 1, fmt_usage(r["usage"])))
            if args.delay:
                time.sleep(args.delay)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            fut_to_idx = {ex.submit(one, i): i for i in range(args.rounds)}
            for fut in as_completed(fut_to_idx):
                indexed[fut_to_idx[fut]] = fut.result()
                done += 1
                if done % 10 == 0 or done == args.rounds:
                    print("  进度: %d/%d" % (done, args.rounds))

    # 整理成按序的明细行
    rows = []
    for i in range(args.rounds):
        r = indexed.get(i, {"error": "missing"})
        if "error" in r:
            rows.append({"idx": i, "status": "error", "error": r["error"],
                         "input": 0, "cache_read": 0, "cache_creation": 0,
                         "output": 0, "hit": False, "ttft_ms": "", "total_ms": ""})
            continue
        u = r["usage"]
        cr = u.get("cache_read_input_tokens") or 0
        rows.append({
            "idx": i, "status": "ok", "error": "",
            "input": u.get("input_tokens") or 0,
            "cache_read": cr,
            "cache_creation": u.get("cache_creation_input_tokens") or 0,
            "output": u.get("output_tokens") or 0,
            "hit": cr > 0,
            "ttft_ms": round(r["ttft"] * 1000, 0) if r["ttft"] is not None else "",
            "total_ms": round(r["total"] * 1000, 0),
        })

    ok_rows = [r for r in rows if r["status"] == "ok"]
    n = len(ok_rows)
    nerr = len(rows) - n
    hits = sum(1 for r in ok_rows if r["hit"])
    total_cache = sum(r["cache_read"] for r in ok_rows)
    # 总输入要含缓存部分：Anthropic 的 input_tokens 不包括 cache_read/cache_creation
    total_input = sum(r["input"] + r["cache_read"] + r["cache_creation"] for r in ok_rows)

    summary = {
        "n": n, "nerr": nerr, "hits": hits,
        "req_hit_rate": (100.0 * hits / n) if n else 0.0,
        "token_hit_rate": (100.0 * total_cache / total_input) if total_input else 0.0,
        "total_input": total_input, "total_cache": total_cache,
        "avg_input": (total_input / n) if n else 0.0,
        "avg_cache": (total_cache / n) if n else 0.0,
    }

    print("\n===== 缓存命中率 =====")
    print("有效请求: %d   失败: %d" % (n, nerr))
    if n:
        print("请求命中率(cache_read>0): %d/%d = %.1f%%" % (hits, n, summary["req_hit_rate"]))
        if total_input:
            print("token 命中率(cache_read/总输入): %d/%d = %.1f%%"
                  % (total_cache, total_input, summary["token_hit_rate"]))
        print("平均 input/req=%.0f   平均 cache_read/req=%.0f"
              % (summary["avg_input"], summary["avg_cache"]))

    if args.html:
        meta = {
            "model": args.model, "url": display_url(args.url),
            "concurrency": args.concurrency, "warmup": args.warmup,
            "prefix_tokens": args.prefix_tokens, "max_tokens": args.max_tokens,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_cache_html(args.html, meta, rows, summary)
        print("\nHTML 报告已写入: %s" % args.html)


def _nice_ceil(v):
    """把上限取整到 1/2/2.5/5/10 × 10^n，让 y 轴刻度好看。"""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def _fmt_tick(v):
    """刻度文本。延迟图的量程常在 0~1s，直接 round 会把 5 档刻度全压成 0/1，
    看着像坏了 —— 小于 10 时保留一位小数。"""
    if v >= 1000:
        return ("%g" % (round(v) / 1000.0)) + "k"
    if v < 10:
        return ("%.1f" % v).rstrip("0").rstrip(".") or "0"
    return str(int(round(v)))


def _svg_line_chart(title, x_labels, series, width=960, height=300):
    """纯 SVG 折线图（无 JS / 无外部依赖）。series=[{name,color,values}]，values 可含 None。"""
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 36
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(x_labels)
    all_vals = [v for s in series for v in s["values"] if v is not None]
    ymax = _nice_ceil(max(all_vals) if all_vals else 1)

    def px(i):
        return pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)

    def py(v):
        return pad_t + ph * (1 - v / ymax)

    p = ['<text x="%g" y="18" class="ct">%s</text>' % (pad_l, html_lib.escape(title))]
    ticks = 5
    for t in range(ticks + 1):
        val = ymax * t / ticks
        yy = py(val)
        p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>' % (pad_l, yy, pad_l + pw, yy))
        p.append('<text x="%g" y="%g" class="yl">%s</text>' % (pad_l - 8, yy + 4, _fmt_tick(val)))
    step = max(1, n // 12)
    for i in range(0, n, step):
        p.append('<text x="%g" y="%g" class="xl">%s</text>'
                 % (px(i), height - 12, html_lib.escape(str(x_labels[i]))))
    for s in series:
        seg, segs = [], []
        for i, v in enumerate(s["values"]):
            if v is None:
                if seg:
                    segs.append(seg)
                    seg = []
            else:
                seg.append((px(i), py(v)))
        if seg:
            segs.append(seg)
        for sg in segs:
            pts = " ".join("%g,%g" % (x, y) for x, y in sg)
            p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pts, s["color"]))
        for i, v in enumerate(s["values"]):
            if v is not None:
                p.append('<circle cx="%g" cy="%g" r="2.5" fill="%s"/>' % (px(i), py(v), s["color"]))
    svg = ('<svg viewBox="0 0 %d %d" class="chart" preserveAspectRatio="xMidYMid meet">%s</svg>'
           % (width, height, "".join(p)))
    leg = " ".join('<span class="lg"><i style="background:%s"></i>%s</span>'
                   % (s["color"], html_lib.escape(s["name"])) for s in series)
    return '<div class="panel"><div class="legendrow">%s</div>%s</div>' % (leg, svg)


def write_cache_html(path, meta, rows, summary):
    """生成自包含的缓存命中率 HTML 报告（折线图，无外部依赖）。"""
    def esc(x):
        return html_lib.escape(str(x))

    def col(key):
        out = []
        for r in rows:
            if r["status"] == "error":
                out.append(None)
            else:
                v = r.get(key)
                out.append(None if v == "" or v is None else float(v))
        return out

    x_labels = [str(r["idx"] + 1) for r in rows]
    chart_tokens = _svg_line_chart(
        "缓存 token (input vs cache_read，每请求)", x_labels,
        [{"name": "input", "color": "#3b82f6", "values": col("input")},
         {"name": "cache_read 命中", "color": "#22c55e", "values": col("cache_read")}])
    chart_latency = _svg_line_chart(
        "延迟 (毫秒，每请求)", x_labels,
        [{"name": "TTFT 首字", "color": "#6366f1", "values": col("ttft_ms")},
         {"name": "total 总耗时", "color": "#f59e0b", "values": col("total_ms")}])

    miss_idx = [r["idx"] + 1 for r in rows if r["status"] == "ok" and not r["hit"]]
    miss_note = ("未命中请求: #%s" % ", #".join(map(str, miss_idx))) if miss_idx else "全部命中"

    cards = [
        ("请求命中率", "%.1f%%" % summary["req_hit_rate"], "%d/%d 命中" % (summary["hits"], summary["n"])),
        ("token 命中率", "%.1f%%" % summary["token_hit_rate"],
         "%d/%d tok" % (summary["total_cache"], summary["total_input"])),
        ("有效/失败", "%d / %d" % (summary["n"], summary["nerr"]), "请求数"),
        ("平均 cache_read", "%.0f" % summary["avg_cache"], "平均 input %.0f" % summary["avg_input"]),
    ]
    card_html = "".join(
        '<div class="card"><div class="card-t">%s</div><div class="card-v">%s</div>'
        '<div class="card-s">%s</div></div>' % (esc(t), esc(v), esc(s))
        for t, v, s in cards)

    meta_html = (
        "模型 <b>%s</b> &nbsp;|&nbsp; 端点 %s &nbsp;|&nbsp; 并发 %d &nbsp;|&nbsp; "
        "预热 %d &nbsp;|&nbsp; 前缀~%d tok &nbsp;|&nbsp; max_tokens %d &nbsp;|&nbsp; %s"
        % (esc(meta["model"]), esc(meta["url"]), meta["concurrency"], meta["warmup"],
           meta["prefix_tokens"], meta["max_tokens"], esc(meta["time"])))

    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>缓存命中率报告 - %(model)s</title>
<style>
  body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       margin:0;background:#f5f6f8;color:#1f2329}
  .wrap{max-width:1040px;margin:0 auto;padding:24px}
  h1{font-size:20px;margin:0 0 4px}
  .meta{color:#6b7280;font-size:12px;margin-bottom:20px}
  .note{font-size:12px;color:#6b7280;margin-bottom:16px}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
  .card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px}
  .card-t{font-size:12px;color:#6b7280}
  .card-v{font-size:26px;font-weight:700;margin:4px 0;color:#111827}
  .card-s{font-size:12px;color:#9ca3af}
  .panel{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin-bottom:16px}
  .chart{width:100%%;height:auto;display:block}
  .ct{font-size:13px;font-weight:600;fill:#374151}
  .grid{stroke:#eef0f2;stroke-width:1}
  .yl{font-size:11px;fill:#9ca3af;text-anchor:end}
  .xl{font-size:11px;fill:#9ca3af;text-anchor:middle}
  .legendrow{margin-bottom:8px;font-size:12px;color:#6b7280}
  .lg{margin-right:14px}
  .lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
</style></head>
<body><div class="wrap">
<h1>缓存命中率报告</h1>
<div class="meta">%(meta)s</div>
<div class="cards">%(cards)s</div>
<div class="note">%(miss)s &nbsp;|&nbsp; 注：input 每请求基本恒定是设计如此——缓存命中要求前缀逐字节相同。</div>
%(chart_tokens)s
%(chart_latency)s
</div></body></html>
""" % {
        "model": esc(meta["model"]),
        "meta": meta_html,
        "cards": card_html,
        "miss": esc(miss_note),
        "chart_tokens": chart_tokens,
        "chart_latency": chart_latency,
    }

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def _fmt_dur(s):
    if s is None:
        return "—"
    return ("%.1fs" % s) if s >= 10 else ("%.0fms" % (s * 1000))


def _ptable(rows):
    head = ('<tr><th>指标</th><th>p50</th><th>p90</th><th>p99</th>'
            '<th>mean</th><th>min</th><th>max</th></tr>')
    body = []
    for label, vals, unit, star in rows:
        def f(v):
            return ("%.1fs" % v) if unit == "s" else ("%.0fms" % (v * 1000))
        if not vals:
            body.append('<tr><td>%s</td><td colspan="6">—</td></tr>' % label)
            continue
        cls = ' class="star"' if star else ''
        body.append('<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                    '<td>%s</td><td>%s</td><td>%s</td></tr>'
                    % (cls, label, f(pct(vals, 50)), f(pct(vals, 90)), f(pct(vals, 99)),
                       f(statistics.mean(vals)), f(min(vals)), f(max(vals))))
    return '<table class="pt"><thead>%s</thead><tbody>%s</tbody></table>' % (head, "".join(body))


def _svg_bar_timeline(cache, width=470, height=220):
    pts = cache["points"]
    pad_l, pad_r, pad_t, pad_b = 38, 8, 10, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    dur = max(cache["duration"], 1)
    lat = [p["latency"] for p in pts if p.get("latency")]
    ymax = _nice_ceil(max(lat) if lat else 1)
    bw = max(2.0, pw / max(len(pts), 1) * 0.6)
    p = []
    for t in range(5):
        yy = pad_t + ph * (1 - t / 4.0)
        p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" class="grid"/>' % (pad_l, yy, pad_l + pw, yy))
        p.append('<text x="%g" y="%g" class="yl">%s</text>' % (pad_l - 6, yy + 3, _fmt_tick(ymax * t / 4.0)))
    for m in range(0, int(dur // 60) + 1):
        xx = pad_l + pw * (m * 60.0 / dur)
        p.append('<text x="%g" y="%g" class="xl">%dm</text>' % (xx, height - 9, m))
    for pt in pts:
        x = pad_l + pw * (pt["t"] / dur) - bw / 2
        if pt.get("err"):
            # 错误请求画红色底标，不然网关一段 5xx 在图上像"没发请求"
            p.append('<rect x="%g" y="%g" width="%g" height="6" fill="#b3261e"/>'
                     % (x, pad_t + ph - 6, bw))
            continue
        if not pt.get("latency"):
            continue
        h = ph * (pt["latency"] / ymax)
        if pt.get("cold"):
            color = "#8fa8b8"      # 首发冷启动（刻意 miss，写缓存）
        elif not pt["hit"]:
            color = "#cec7b6"      # 未命中
        elif pt.get("ratio", 1.0) < 0.8:
            color = "#d9a35e"      # 部分命中（cache_read 占比 < 80%）
        else:
            color = "#b8560f"      # 命中
        p.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s"/>'
                 % (x, pad_t + ph - h, bw, h, color))
    return '<svg viewBox="0 0 %d %d" class="chart">%s</svg>' % (width, height, "".join(p))


def _metric_json(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(values),
            "p50_ms": round(pct(values, 50) * 1000, 2),
            "p90_ms": round(pct(values, 90) * 1000, 2),
            "p99_ms": round(pct(values, 99) * 1000, 2),
            "mean_ms": round(statistics.mean(values) * 1000, 2),
            "min_ms": round(ordered[0] * 1000, 2),
            "max_ms": round(ordered[-1] * 1000, 2)}


def report_json(meta, bench, cache, roll, lerr):
    """机器可读结果，结构对齐 responses_bench 的 --json（字段名一致，便于服务端统一处理）。"""
    classes = {}
    for e in lerr:
        name, _ = classify_error(e)
        classes[name] = classes.get(name, 0) + 1
    out = {
        "meta": meta,
        "counts": {"ok": bench["ok"], "fail": bench["fail"]},
        "error_classes": classes,
        "error_samples": [str(e)[:300] for e in lerr[:5]],
        "throughput": {"wall_s": round(bench["wall"], 3),
                       "req_per_s": round(bench["req_throughput"], 3),
                       "output_tok_per_s": round(bench["out_throughput"], 1),
                       "decode_tps": round(bench["tps_mean"], 2)},
        "latency": {"first_text": _metric_json(bench["ttft"]),
                    "tpot_visible": _metric_json(bench["tpot"]),
                    "total": _metric_json(bench["total"])},
    }
    if cache is not None:
        out["cache"] = {k: v for k, v in cache.items() if k != "points"}
        out["cache"]["rows"] = cache["points"]
    if roll:
        out["roll"] = dict(roll)
    return out


def run_report(args):
    # 采样发数优先：内部仍按时长驱动（定时器好写），但对外允许直接说要多少发。
    # 换算用 samples×interval：循环条件是 elapsed < duration，t=0 也发一发，
    # 所以 N 发恰好需要 N×interval 秒（最后一发在 (N-1)×interval 时刻发出）。
    if getattr(args, "cache_samples", 0) > 0:
        args.cache_duration = args.cache_samples * max(1, args.cache_interval)
    prefix = build_prefix(args.prefix_tokens)
    print("[report] 首字压测(并发%d, %d请求)%s"
          % (args.concurrency, args.requests,
             "" if args.no_cache else " + 缓存持久性(%ds, 每%ds一发)" % (args.cache_duration, args.cache_interval)))
    # 进度计划：preflight → load → cache（samples>0 才有）→ roll（turns>0 才有）
    _plan = (([] if args.no_preflight else ["preflight"]) + ["load"]
             + (["cache"] if not args.no_cache else [])
             + (["roll"] if args.roll_turns > 0 else []))
    progress_plan(_plan)

    def one(i):
        payload = {
            "model": args.model, "max_tokens": args.max_tokens, "stream": True,
            "system": system_block(prefix, cache=True),
            "messages": [{"role": "user", "content": "压测请求 #%d，简短回答。" % i}],
            # 压测：每请求不同完整 user_id（同 base、变 account/session）→ 亲和路由把负载分散到
            # 各渠道（模拟多用户高并发），同时 base 不变 → 不触发网关唯一用户限额/封禁
            "metadata": {"user_id": _claude_user_id()},
        }
        return send_stream(args.url, args.token, apply_cc_fields(payload, args), args.timeout)

    # 预检：report 要跑压测 + 几分钟缓存采样，端点不通时白等最久，更值得先探一次
    if not args.no_preflight:
        progress("preflight", 0, 1)
        if not preflight(args, lambda: one(-1)):
            return
        progress("preflight", 1, 1)

    # ① 首字压测：并发
    # 真实上游普遍有 RPM 限额（实测腾讯 tokenhub 是 60），不限速直接甩 100 个并发
    # 会撞一墙 429——量到的不是首字延迟，是限流器的响应速度。--request-rate 给一个
    # 到达速率上限，池子仍是 --concurrency，实际在飞的请求数由两者中更紧的那个决定。
    rate = float("inf") if str(args.request_rate) == "inf" else float(args.request_rate)
    print("① 首字压测：并发 %d%s ..."
          % (args.concurrency,
             "" if rate == float("inf") else "，限速 %.2f req/s (≈%.0f RPM)" % (rate, rate * 60)))
    load = {}
    t0 = time.time()
    progress("load", 0, args.requests)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fmap = {}
        for i in range(args.requests):
            fmap[ex.submit(one, i)] = i
            if rate != float("inf") and i < args.requests - 1:
                time.sleep(random.expovariate(rate))   # Poisson 到达
        done = 0
        for fut in as_completed(fmap):
            load[fmap[fut]] = fut.result()
            done += 1
            progress("load", done, args.requests)
            if done % 20 == 0 or done == args.requests:
                print("   %d/%d" % (done, args.requests))
    load_wall = time.time() - t0

    ttfts, tpots, totals, lerr = [], [], [], []
    lout = 0
    for i in range(args.requests):
        r = load.get(i, {"error": "missing"})
        if "error" in r:
            lerr.append(r["error"])
            continue
        u = r["usage"]
        lout += u.get("output_tokens") or 0
        if r["ttft"] is not None:
            ttfts.append(r["ttft"])
        totals.append(r["total"])
        if r["tpot"] is not None:
            tpots.append(r["tpot"])
    lok = len(totals)
    bench = {
        "ok": lok, "fail": len(lerr), "wall": load_wall,
        "req_throughput": lok / load_wall if load_wall else 0,
        "out_throughput": lout / load_wall if load_wall else 0,
        # 解码速度 tok/s = 1 / 中位 TPOT（用中位避免 token 爆发到达的离群值）
        "tps_mean": (1.0 / statistics.median(tpots)) if tpots else 0,
        "ttft": ttfts, "tpot": tpots, "total": totals,
    }

    cache = None
    roll_ctx = None
    if not args.no_cache:
        # ② 缓存持久性：用独立的新前缀（避免被上面的压测焐热），按固定节奏连发，
        #    第一发是冷的(miss)，之后看缓存能否随时间保持命中
        # 标记放最前面：上游是前缀自动缓存，标记必须在开头才能让前缀从第0字符就分叉、真冷启动
        cache_prefix = ("[persistence-test-%d]\n\n" % os.getpid()) + prefix

        def cache_one(i):
            payload = {
                "model": args.model, "max_tokens": args.max_tokens, "stream": True,
                # 只放 CLI 身份前缀（过门卫），大块仍在 messages
                "system": cc_system_preamble(),
                # 大块必须放 messages 才会被读回，放 system 恒 0% —— 详见 cache_messages()
                "messages": cache_messages(cache_prefix, "缓存持久性 #%d，简短回答。" % i),
                # 缓存测试：全程同一个稳定 user_id → 亲和路由到同一渠道 → 缓存能持续命中
                "metadata": {"user_id": _RUN_USER_ID},
            }
            return send_stream(args.url, args.token, apply_cc_fields(payload, args), args.timeout)

        interval = max(1, args.cache_interval)  # 防 0：cache-interval=0 会退化成全程无间隔洪泛
        print("② 缓存持久性：先同步冷启动写入 2 次，再每 %ds 一发、持续 %ds（独立新前缀）..."
              % (interval, args.cache_duration))

        def _to_point(fire_t, r, cold=False):
            if "error" in r:
                # 留下错误原文：只记 err=True 的话，缓存阶段全挂时只能看到"错误 N"，
                # 分不清是零散抖动还是某刻起被配额/限流闩死（后者性质完全不同）
                return {"t": fire_t, "latency": None, "hit": False, "err": True, "cold": cold,
                        "cache_read": 0, "cache_creation": 0, "input": 0, "ratio": 0.0,
                        "msg": r["error"]}
            uu = r["usage"]
            cr = uu.get("cache_read_input_tokens") or 0
            cc = uu.get("cache_creation_input_tokens") or 0
            inp = uu.get("input_tokens") or 0
            den = cr + cc + inp
            return {"t": fire_t, "latency": r["total"], "hit": cr > 0, "cold": cold,
                    "cache_read": cr, "cache_creation": cc, "input": inp,
                    "ratio": (float(cr) / den) if den else 0.0}

        # 冷启动同步写入：先等它把缓存真正建好，再开始定时采样。否则慢网关（首字十几秒）下，
        # 后面几发会在缓存尚未写入时就抢跑、全部 miss，把正常缓存误判成"不可靠/逐出重写"。
        # 写 2 次而不是 1 次：实测缓存写入对后续请求可见有一发延迟（第1发 creation>0/read=0，
        # 第2发仍 creation>0/read=0，第3发才开始 read>0），只写 1 次会让采样首发照样 miss。
        pts = [_to_point(0.0, cache_one(-1), cold=True),
               _to_point(0.0, cache_one(-2), cold=True)]

        pending = []
        sched_start = time.time()
        if getattr(args, "roll_turns", 0) > 0:
            # ③ 的预热放进采样窗口末尾的后台线程做，省掉 ②→③ 之间的干等；
            # 用独立前缀，不会碰 ② 的缓存，也只在最后 30 秒启动，离 ③ 的第 1 轮远小于缓存有效期
            args.tools = args.roll_tools
            args.turn_tokens = args.roll_turn_tokens
            roll_ctx = roll_prepare(args, verbose=False, turns=args.roll_turns, tag="report-roll")

            def _warm_bg():
                try:
                    roll_warm(args, roll_ctx, verbose=False, delay=0)
                except Exception:  # noqa
                    pass

            roll_ctx["warm_thread"] = threading.Timer(max(0.0, args.cache_duration - ROLL_WARM_LEAD_S), _warm_bg)
            roll_ctx["warm_thread"].daemon = True
            roll_ctx["warm_thread"].start()
        tick = 0
        _cache_total = max(1, int(math.ceil(args.cache_duration / float(interval))))
        _cache_done = [0]
        progress("cache", 0, _cache_total)

        def _cache_cb(_f):
            with _PROGRESS_LOCK:
                _cache_done[0] += 1
                d = _cache_done[0]
            progress("cache", d, _cache_total)

        with ThreadPoolExecutor(max_workers=32) as ex:
            while time.time() - sched_start < args.cache_duration:
                fire_t = time.time() - sched_start
                _fut = ex.submit(cache_one, tick)
                _fut.add_done_callback(_cache_cb)
                pending.append((fire_t, _fut))
                tick += 1
                nxt = sched_start + tick * interval
                time.sleep(max(0, nxt - time.time()))
            print("   等待 %d 个缓存请求收尾 ..." % len(pending))
            for fire_t, fut in pending:
                pts.append(_to_point(fire_t, fut.result()))

        # 计分排除首发冷启动（必 miss 是设计使然）和错误请求
        scored = [p for p in pts[1:] if not p.get("err")]
        chits = sum(1 for p in scored if p["hit"])
        cn = len(scored)
        nerr = sum(1 for p in pts if p.get("err"))
        first_hit = next((p["t"] for p in pts if p["hit"]), None)
        tok_num = sum(p["cache_read"] for p in scored)
        tok_den = sum(p["cache_read"] + p["cache_creation"] + p["input"] for p in scored)
        # miss 且 cache_creation>0 = 缓存被逐出后重写；两者都 0 = 上游没有缓存迹象
        rewrites = sum(1 for p in scored if not p["hit"] and p["cache_creation"] > 0)
        dead_miss = sum(1 for p in scored if not p["hit"] and p["cache_creation"] == 0)
        cache = {
            "points": pts, "n": cn, "hits": chits, "nerr": nerr,
            "hit_rate": 100.0 * chits / cn if cn else 0,
            "token_hit_rate": 100.0 * tok_num / tok_den if tok_den else 0,
            "rewrites": rewrites, "dead_miss": dead_miss,
            "first_hit": first_hit, "duration": args.cache_duration, "interval": interval,
        }

    print("\n===== 汇总 =====")
    if ttfts:
        print("TTFT p50=%.0fms p90=%.0fms | 平均TPS=%.1f | 吞吐=%.2f req/s"
              % (pct(ttfts, 50) * 1000, pct(ttfts, 90) * 1000, bench["tps_mean"], bench["req_throughput"]))
    if cache:
        print("缓存命中(除首发冷启动) %d/%d = %.0f%%   token口径 %.0f%%   首次命中=%s"
              % (chits, cn, cache["hit_rate"], cache["token_hit_rate"],
                 ("%.0fs" % first_hit) if first_hit is not None else "无"))
        if cache["rewrites"] or cache["dead_miss"] or nerr:
            print("未命中构成: 逐出重写 %d · 无缓存迹象 %d · 错误 %d"
                  % (cache["rewrites"], cache["dead_miss"], nerr))
        # 有效样本太少时，命中率是拿零星几发算出来的，不能当结论用。
        # 实测教训（2026-08-26）：50 发采样挂了 48 发，只剩 2 发有效，报告照样打出
        # 「100% 命中」——数字本身没错，但拿它下结论是灾难性的。
        total_pts = len(pts)
        if cn < 10 or (total_pts and nerr > total_pts * 0.3):
            print("⚠️  有效样本仅 %d/%d（失败 %d）——上面的命中率不可作为结论，"
                  "先排掉失败原因再重测。" % (cn, total_pts, nerr))
    # 压测阶段的错误样本：只报个数不报原因的话，全挂时（模型名错/403/503）拿不到任何线索，
    # 还得退回 bench 重跑一遍才看得见报错
    if lerr:
        print_error_buckets(lerr)
        print("\n① 压测错误样本(共 %d 条):" % len(lerr))
        for e in lerr[:5]:
            print("  -", e)
    if cache:
        # 缓存阶段同理，并带上发生时刻：连续区间 = 配额/限流闩死，零散 = 抖动
        cerr = [p for p in pts if p.get("err")]
        if cerr:
            print("\n② 缓存阶段错误样本(共 %d 条，分布 %.0fs ~ %.0fs):"
                  % (len(cerr), cerr[0]["t"], cerr[-1]["t"]))
            for p in cerr[:5]:
                print("  - t=%.0fs %s" % (p["t"], str(p.get("msg", ""))[:200]))

    # ③ 滚动缓存：多轮对话，看命中区会不会跟着对话长大。
    # 为什么必须单独测：上面②测的是"同一段话隔 8s 重发，缓存能活多久"（持久性），
    # 每一发的聊天记录都是空的。而真实会话是记录越攒越多，那部分能不能也进缓存
    # 完全没被②覆盖 —— 恰恰是长会话最花钱的地方。
    roll = None
    if getattr(args, "roll_turns", 0) > 0:
        print("\n③ 滚动缓存：%d 轮真实对话（历史逐轮累积，看命中区增长）..." % args.roll_turns)
        # roll_collect 读的是 args.tools / args.turn_tokens，report 用 --roll-* 前缀避免和别的模式撞名
        args.tools = args.roll_tools
        args.turn_tokens = args.roll_turn_tokens
        rows = roll_collect(args, verbose=False, turns=args.roll_turns, tag="report-roll",
                            ctx=roll_ctx, warm_delay=0)
        if rows:
            roll = roll_stats(rows)
            print("   命中区 %s → %s   捕获 %.0f%%   省下 %.1f%%   %s"
                  % ("{:,}".format(roll["reads"][0]), "{:,}".format(roll["reads"][-1]),
                     roll["capture"] * 100, roll["saved"], roll["verdict"]))
        else:
            print("   跳过（首轮即失败）")

    meta = {
        "model": args.model, "url": display_url(args.url), "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "prefix_tokens": args.prefix_tokens,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    eff = (getattr(args, "effort", None) or "").strip().lower()
    if eff not in ("none", "off", ""):
        meta["effort"] = eff
    write_report_html(args.html, meta, bench, cache, roll)
    print("HTML 报告: %s" % args.html)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report_json(meta, bench, cache, roll, lerr), f, ensure_ascii=False, indent=2)
        print("JSON 已写入: %s" % args.json)


def _roll_panel(roll, esc):
    """③ 滚动缓存板块：双折线（总输入 vs 命中区）+ 明细表 + 判定。"""
    if not roll:
        return ""
    rows = roll["rows"]
    W, H, PL, PR, PT, PB = 480, 170, 52, 10, 12, 26
    iw, ih = W - PL - PR, H - PT - PB
    ymax = max(max(roll["totals"]), 1) * 1.08
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
    xlab = "".join('<text class="xl" x="%.1f" y="%d">%d</text>'
                   % (xy(i, 0)[0], H - 8, r["turn"])
                   for i, r in enumerate(rows) if i % max(1, len(rows) // 8) == 0)
    dots = "".join('<circle cx="%.1f" cy="%.1f" r="2.5" fill="#0d7268"/>'
                   % xy(i, v) for i, v in enumerate(roll["reads"]))
    svg = ('<svg class="chart" viewBox="0 0 %d %d">%s'
           '<polyline fill="none" stroke="#a9a294" stroke-width="1.5" stroke-dasharray="4 3" points="%s"/>'
           '<polyline class="draw" fill="none" stroke="#0d7268" stroke-width="2" points="%s"/>%s%s</svg>'
           % (W, H, grid, poly(roll["totals"]), poly(roll["reads"]), dots, xlab))

    body = "".join(
        "<tr><td>第%d轮</td><td>%s</td><td>%s</td><td>%s</td><td>%.0f%%</td></tr>"
        % (r["turn"], "{:,}".format(r["total"]), "{:,}".format(r["input"]),
           "{:,}".format(r["read"]),
           (r["read"] / r["total"] * 100) if r["total"] else 0)
        for r in rows)
    table = ('<table class="pt"><tr><th>轮次</th><th>总输入</th><th>付费input</th>'
             '<th>命中区</th><th>命中率</th></tr>%s</table>' % body)

    tagcls = {"ok": "ok", "warn": "warn", "bad": "bad"}[roll["level"]]
    lgd = ('<i style="background:#a9a294"></i>总输入(上下文全量)'
           '<i style="background:#0d7268"></i>命中区(缓存读回，不花钱)')
    note = ("第%d轮起统计 · 历史增长 %+d tok · 命中区捕获 %.0f%% · "
            "累计省下 %.1f%%（%s → 实付 %s）"
            % (roll["warm_turn"], roll["hist_growth"], roll["capture"] * 100,
               roll["saved"], "{:,}".format(roll["sum_total"]),
               "{:,}".format(roll["sum_input"])))
    # 左图右表：板块是整宽的，图若独占一行会被 width:100% 拉到几倍高，
    # 纵轴刻度被扯得老远、读数反而更费劲。
    return ('<div class="panel"><h2>③ 滚动缓存 · %d 轮真实对话 · 历史逐轮累积'
            '<span class="tag %s">%s</span></h2>'
            '<div class="rollgrid"><div>%s<div class="lgd">%s</div>'
            '<div class="tp">%s</div></div><div>%s</div></div></div>'
            % (len(rows), tagcls, esc(roll["verdict"][:24]), svg, lgd, esc(note), table))


def write_report_html(path, meta, bench, cache, roll=None):
    """单页自包含 HTML 报告。

    视觉方向：工程绘图纸（暖白纸底 + 淡网格 + 墨色等宽 + 烧橙/深青双色）。
    读者是工程师，信息密度高、要求精确对齐，图纸这套语言比通用 dashboard 更贴合，
    也避开了"白底紫渐变"那种烂大街的模板脸。

    卡片区 5 个正交读数。缓存拆成两张但语义不重叠 ——
      缓存命中率   静态口径：同一段话重发，命中没命中（请求数占比）
      增量命中率   Incremental Cache Coverage：新长出来的对话历史被缓存接住了多少
    这是两件不同的事：前者高不代表后者高，长会话烧不烧钱看的是后者。
    """
    def esc(x):
        return html_lib.escape(str(x))

    ttft = bench["ttft"]
    total_req = bench["ok"] + bench["fail"]
    p50 = _fmt_dur(pct(ttft, 50)) if ttft else "—"
    p90 = _fmt_dur(pct(ttft, 90)) if ttft else "—"

    cards = [
        ("TTFT p50", p50, "并发 %d · p90 %s" % (meta["concurrency"], p90), "am"),
        ("解码速度", "%.0f" % bench["tps_mean"], "tok/s 每流", ""),
        # 别写成「错误 0/50」——分子分母同时出现时，读者会本能地理解成"50 个里成了 0 个"，
        # 正好把最好的结果读成最坏的。改成成功率打头：100% 不可能被误读，
        # 失败次数退到副标，需要时照样看得见。
        ("成功率", "%.0f%%" % (bench["ok"] * 100.0 / total_req if total_req else 0),
         ("%d 次请求 · 零失败" % total_req) if bench["fail"] == 0
         else ("%d 次请求 · 失败 %d 次" % (total_req, bench["fail"])),
         "bad" if bench["fail"] else ""),
    ]
    if cache:
        cards.append(("缓存命中率", "%.0f%%" % cache["hit_rate"],
                      "%d/%d 请求 · token 口径 %.0f%%" % (cache["hits"], cache["n"], cache["token_hit_rate"]),
                      "cy"))
    if roll:
        cards.append(("增量命中率", "%.0f%%" % (roll["capture"] * 100),
                      "%d 轮滚动 · 累计省下 %.0f%%" % (len(roll["rows"]), roll["saved"]),
                      "cy"))
    card_html = "".join(
        '<div class="card%s" style="--i:%d"><div class="card-t">%s</div>'
        '<div class="card-v">%s</div><div class="card-s">%s</div></div>'
        % ((" " + cls if cls else ""), i, esc(t), esc(v), esc(sub))
        for i, (t, v, sub, cls) in enumerate(cards))

    ptable = _ptable([
        ("TTFT 首字", ttft, "s", True),
        ("TPOT/token", bench["tpot"], "ms", False),
        ("总耗时", bench["total"], "s", False),
    ])
    stable = bool(ttft) and pct(ttft, 50) > 0 and (pct(ttft, 90) / pct(ttft, 50) < 2.0)
    throughput = ("请求吞吐 <b>%.2f</b> req/s &nbsp;·&nbsp; 输出吞吐 <b>%.0f</b> tok/s &nbsp;·&nbsp; "
                  "总墙钟 <b>%.1fs</b>" % (bench["req_throughput"], bench["out_throughput"], bench["wall"]))
    tag_stable = ('<span class="tag ok">稳定</span>' if stable
                  else '<span class="tag warn">波动</span>')
    cache_panel = ""
    cache_foot = ""
    if cache:
        reliable = cache["hit_rate"] >= 80
        tag_cache = ('<span class="tag ok">可靠 %.0f%%</span>' % cache["hit_rate"] if reliable
                     else '<span class="tag bad">不可靠 %.0f%%</span>' % cache["hit_rate"])
        first_hit = ("%.0fs" % cache["first_hit"]) if cache["first_hit"] is not None else "无"
        lgd = ("纵轴=延迟(s) 横轴=分钟 ·<i style=\"background:#b8560f\"></i>命中"
               "<i style=\"background:#d9a35e\"></i>部分<i style=\"background:#cec7b6\"></i>未命中"
               "<i style=\"background:#8fa8b8\"></i>冷启动<i style=\"background:#b3261e\"></i>错误"
               " · 请求 %.0f%% / token %.0f%% · 首次命中 %s"
               % (cache["hit_rate"], cache["token_hit_rate"], first_hit))
        miss_mix = ""
        if cache["rewrites"] or cache["dead_miss"]:
            miss_mix = "（未命中: 逐出重写 %d · 无缓存迹象 %d）" % (cache["rewrites"], cache["dead_miss"])
        cache_panel = ('<div class="panel"><h2>② 缓存 · 持久性 · 连续 %ds</h2>%s<div class="lgd">%s</div></div>'
                       % (cache["duration"], _svg_bar_timeline(cache), lgd))
        cache_foot = (" · 缓存: 连续 %ds/每 %ds 一发, 前缀~%d tok, 首次命中 %s · 缓存命中 %s%s"
                      % (cache["duration"], cache["interval"], meta["prefix_tokens"],
                         first_hit, tag_cache, miss_mix))
    footer = ("并发 %d × %d 请求, max_tokens=%d, 流式%s · 结论: 并发性能 %s"
              % (meta["concurrency"], total_req, meta["max_tokens"], cache_foot, tag_stable))
    sub = "%s · %s · %s" % (esc(meta["url"]), esc(meta["model"]), esc(meta["time"]))

    html = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>%(model)s 压测报告</title><style>
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
 .rule{height:3px;background:linear-gradient(90deg,var(--am),var(--cy) 38%%,var(--line) 78%%)}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
        gap:1px;background:var(--line);border:1px solid var(--line);
        border-top:none;margin:0 0 22px}
 .card{background:var(--surface);padding:15px 17px 14px;position:relative;
       animation:rise .5s cubic-bezier(.2,.7,.3,1) both;animation-delay:calc(var(--i)*65ms)}
 .card::after{content:"";position:absolute;left:0;top:0;width:100%%;height:2px;background:var(--line2)}
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
 table.pt{width:100%%;border-collapse:collapse;font-variant-numeric:tabular-nums}
 .pt th,.pt td{padding:5.5px 8px;text-align:right;font-size:11.5px;
               border-bottom:1px solid var(--line)}
 .pt th:first-child,.pt td:first-child{text-align:left;color:var(--dim)}
 .pt th{color:var(--dim);font-weight:500;font-size:10px;letter-spacing:1px}
 .pt tr:last-child td{border-bottom:none}
 .pt tr.star td{color:var(--am)}
 .pt tr:hover td{background:#f7f3e9}
 .tp{font-size:11px;color:var(--dim);margin-top:11px;line-height:1.7}
 .tp b{color:var(--ink);font-weight:600}
 .chart{width:100%%;height:auto;display:block;max-height:250px;overflow:visible}
 .grid{stroke:#eae5d8;stroke-width:1}
 .yl{fill:#96907f;font-size:9.5px;text-anchor:end;font-family:var(--mono)}
 .xl{fill:#96907f;font-size:9.5px;text-anchor:middle;font-family:var(--mono)}
 .draw{stroke-dasharray:4000;stroke-dashoffset:4000;animation:dash 1.5s .45s ease-out forwards}
 @keyframes dash{to{stroke-dashoffset:0}}
 .lgd{font-size:10.5px;color:var(--dim);margin-top:9px;line-height:1.8}
 .lgd i{display:inline-block;width:8px;height:8px;margin:0 4px 0 10px;vertical-align:middle}
 .footer{font-size:10.5px;color:var(--dim);margin-top:18px;padding-top:15px;
         border-top:1px solid var(--line);line-height:1.9}
 .tag{display:inline-block;padding:2px 9px;font-size:10px;letter-spacing:.6px;
      border:1px solid currentColor}
 .tag.ok{color:var(--cy)} .tag.warn{color:var(--am)} .tag.bad{color:var(--bad)}
 @media(max-width:880px){.grid2,.rollgrid{grid-template-columns:1fr}.sub{text-align:left}}
</style><body><div class="wrap">
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
        "model": esc(meta["model"]), "sub": sub, "cards": card_html,
        "conc": meta["concurrency"], "ptable": ptable, "throughput": throughput,
        "cachepanel": cache_panel, "footer": footer,
        "rollpanel": _roll_panel(roll, esc),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    p = argparse.ArgumentParser(description="Claude /v1/messages 压测 + TTFT/TPOT/ITL + 缓存测试")
    p.add_argument("--url", required=True, help="完整端点，如 https://host/v1/messages")
    # 密钥改为优先读环境变量 BENCH_API_KEY，避免出现在进程列表里；--token 仍兼容
    p.add_argument("--token", default=os.environ.get("BENCH_API_KEY", ""),
                   help="密钥；推荐用环境变量 BENCH_API_KEY 传入，管理员固定渠道可加 -<渠道ID> 后缀")
    p.add_argument("--profile", choices=("claude-code", "generic"), default="claude-code",
                   help="请求画像：claude-code=原行为(伪装 Claude Code)；generic=通用 Messages 兼容接口，"
                        "不带 Claude Code 专用头/system 块，thinking/effort 默认不发送")
    p.add_argument("--model", default="glm-5.2")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="单请求最大输出 token。推理模型的思考与可见正文共用这份预算，"
                        "且部分网关把推理 token 计进 output_tokens 却不下发任何 delta——"
                        "预算太小会在思考结束前耗尽，stop_reason=max_tokens 但一个字都收不到，"
                        "被记成「空响应」假失败。别往下调，原因见手册「测推理模型前必读」")
    p.add_argument("--prefix-tokens", type=int, default=8000, help="system 前缀大小(粗估 token)")
    p.add_argument("--thinking-budget", type=int, default=0,
                   help="[兼容旧写法] 等价于 --thinking enabled:N；0=不使用。"
                        "会自动把 max_tokens 抬到 budget+256")
    p.add_argument("--thinking", default="adaptive",
                   help="thinking 字段：adaptive(默认，对齐真实 Claude Code) / "
                        "enabled:N / enabled / disabled / none(不发送)。"
                        "老兼容层拒 adaptive，Kimi 系拒 disabled，换渠道必测")
    p.add_argument("--effort", default="xhigh",
                   help="output_config.effort：xhigh(默认，对齐真实 Claude Code) / "
                        "high / max / low / none(不发送)。⚠️ 智谱全线只认 low/high/max，"
                        "发 xhigh 会报 [1210]，但那个报错文案说的是「不支持关闭思考」，别被误导")
    p.add_argument("--mid-conv-system", action="store_true",
                   help="在 messages 末尾插一条 role:\"system\" 消息(mid-conversation "
                        "system，2026-04 新 beta)。老实现把 role 当枚举，会直接 400")
    p.add_argument("--no-preflight", action="store_true",
                   help="跳过开压前的单发预检（默认会先探一次，失败就中止，省得错误刷满整轮）")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验(自签名/内网端点)")
    p.add_argument("--user-id", default="",
                   help="固定 metadata.user_id（网关按它做缓存亲和路由到同一渠道）；留空自动生成")

    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="并发/限速压测")
    b.add_argument("--concurrency", type=int, default=10, help="最大并发数")
    b.add_argument("--requests", type=int, default=50, help="总请求数")
    b.add_argument("--request-rate", default="inf",
                   help="目标到达速率 req/s (Poisson)，inf=不限速(仅受并发限制)")
    b.add_argument("--csv", default="", help="导出每请求明细到 CSV 文件")
    b.add_argument("--json", default="", help="机器可读结果 JSON 路径")
    b.set_defaults(func=run_bench)

    rp = sub.add_parser("report", help="一页 HTML 压测报告(首字 + 缓存持久性)")
    rp.add_argument("--concurrency", type=int, default=100, help="首字压测并发数")
    rp.add_argument("--requests", type=int, default=100, help="首字压测总请求数")
    rp.add_argument("--cache-duration", type=int, default=180, help="缓存持久性测试时长(秒)")
    rp.add_argument("--cache-interval", type=int, default=8, help="缓存测试发送间隔(秒)")
    rp.add_argument("--cache-samples", type=int, default=40,
                    help="缓存阶段采样发数(默认 40)。按 samples×interval 反推时长，覆盖 --cache-duration："
                         "40×8s ≈ 320s。人想的是「测40发」而不是「跑320秒」，省得每次手算。"
                         "0=改用 --cache-duration 计时")
    rp.add_argument("--no-cache", action="store_true",
                    help="跳过 ② 缓存持久性阶段（快速档用）")
    rp.add_argument("--roll-turns", type=int, default=12,
                    help="③ 滚动缓存的轮数(默认 12，0=跳过该板块)。②测的是「同一段话能存多久」，"
                         "每发的聊天记录都是空的；③才测「记录越攒越多时能不能也被缓存接住」。"
                         "8 轮足够判定能不能接住；12 轮能多看几轮增长是否稳住，"
                         "要下「能不能长期用」的结论建议 30 轮")
    rp.add_argument("--roll-tools", type=int, default=20,
                    help="③ 滚动测试伪造几个工具定义(真实 Claude Code 的 tools 占请求体 83%%)")
    rp.add_argument("--roll-turn-tokens", type=int, default=500,
                    help="③ 每轮往提问里塞多少 token。默认 500：太小会被缓存块粒度(64~128)淹没")
    rp.add_argument("--request-rate", default="inf",
                    help="① 首字压测的到达速率上限 req/s (Poisson)；inf=不限速。"
                         "上游有 RPM 限额时必须给，否则满屏 429。RPM 60 → 填 0.9")
    rp.add_argument("--html", default="report.html", help="输出 HTML 路径")
    rp.add_argument("--json", default="", help="机器可读结果 JSON 路径")
    rp.set_defaults(func=run_report)

    c = sub.add_parser("cache", help="缓存命中率测试")
    c.add_argument("--rounds", type=int, default=20, help="总请求数(如 100)")
    c.add_argument("--concurrency", type=int, default=1, help="并发数，1=顺序(命中率最高)")
    c.add_argument("--warmup", type=int, default=2,
                   help="预热次数(先写缓存，不计入统计)。默认 2：实测写入对后续请求可见有"
                        "一发延迟，智谱官方前 4 发都是 cache_read=0，预热 1 次不够")
    c.add_argument("--cache-placement", default="messages",
                   choices=("messages", "system", "both"),
                   help="稳定大块/断点放哪：messages(默认，兼容性最好) / system / "
                        "both(先诊断两种放法，识别「只认 messages 断点」这类上游缺陷)")
    c.add_argument("--delay", type=float, default=0, help="顺序模式下每次之间 sleep 秒数")
    c.add_argument("--html", default="", help="导出缓存命中率 HTML 报告到指定文件")
    c.set_defaults(func=run_cache)

    r = sub.add_parser("roll", help="滚动缓存测试(多轮对话，看命中区是否随对话增长)")
    r.add_argument("--turns", type=int, default=12, help="对话轮数(默认 12，和 report 的 --roll-turns 一致)")
    r.add_argument("--warmup", type=int, default=2,
                   help="开跑前用首轮 payload 预热几次(写入对后续可见有延迟，见 run_roll 说明)")
    r.add_argument("--tools", type=int, default=20,
                   help="伪造多少个工具定义塞进 tools。真实 Claude Code 的 tools 占请求体 83%%，"
                        "不带 tools 测出来的省钱比例是虚的。0=不带")
    r.add_argument("--turn-tokens", type=int, default=500,
                   help="每轮往提问里塞多少 token 的填充，让各家增长速率一致便于横向对比。"
                        "0=不填充(默认，最贴近真实形态)")
    r.add_argument("--delay", type=float, default=1.5, help="每轮之间 sleep 秒数")
    r.set_defaults(func=run_roll)

    args = p.parse_args()
    check_url(p, args.url)
    if not args.token:
        p.error("请设置环境变量 BENCH_API_KEY（或传 --token）")
    global _SSL_CTX, _RUN_USER_ID, _RUN_BASE_USER, _PROFILE
    _PROFILE = args.profile
    if _PROFILE == "generic":
        # generic 下 thinking/effort 默认不发送；用户显式传了才发
        if args.thinking == "adaptive":
            args.thinking = "none"
        if args.effort == "xhigh":
            args.effort = "none"
    if args.insecure:
        _SSL_CTX = ssl._create_unverified_context()
    _RUN_BASE_USER = _new_base_user()
    _RUN_USER_ID = args.user_id or _claude_user_id()
    args.func(args)


if __name__ == "__main__":
    main()
