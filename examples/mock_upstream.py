"""本地模拟上游：/v1/messages 与 /v1/responses（SSE）+ /v1/models，无需真实密钥即可体验 api-bench。"""
import hashlib
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEEN = set()
LOCK = threading.Lock()
LOG = []          # 每个请求的 (path, headers dict, body dict)
MODE = {"fail": "", "delay": 0.01, "ttft": None}   # fail: "" | "401echo" | "500"；ttft=(最小,最大)秒，首字前随机等待
SEEN_MSG = set()  # 已见过的「前 k 条消息」累积哈希，用来模拟滚动缓存随对话增长
GROW = 700        # 每多命中一条历史消息，多缓存这么多 token


def _prefix_key(body):
    parts = [json.dumps(body.get("system", ""), ensure_ascii=False)]
    msgs = body.get("messages") or body.get("input") or []
    if msgs:
        parts.append(json.dumps(msgs[0], ensure_ascii=False)[:4000])
    return hashlib.sha1("".join(parts).encode()).hexdigest()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?")[0].endswith("/models"):
            return self._send(200, {"data": [{"id": "mock-a"}, {"id": "mock-b"}]})
        self._send(404, {"error": "nf"})

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if MODE["ttft"]:
            time.sleep(random.uniform(*MODE["ttft"]))
        for ev in events:
            data = ("data: " + json.dumps(ev) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
            self.wfile.flush()
            time.sleep(MODE["delay"])
        self.wfile.write(b"0\r\n\r\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        LOG.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if MODE["fail"] == "401echo":
            tok = self.headers.get("Authorization", "")
            return self._send(401, {"error": {"message": "invalid key " + tok}})
        if MODE["fail"] == "500":
            return self._send(500, {"error": "boom"})
        pk = _prefix_key(body)
        with LOCK:
            hit = pk in SEEN
            SEEN.add(pk)
        big = 6000
        msgs = body.get("messages") or body.get("input") or []
        grown, ok = 0, True
        with LOCK:
            for i in range(2, len(msgs) + 1):
                h = hashlib.sha1((pk + json.dumps(msgs[:i], ensure_ascii=False)).encode()).hexdigest()
                if ok and h in SEEN_MSG:
                    grown += 1
                else:
                    ok = False
                SEEN_MSG.add(h)
        extra = GROW * grown if hit else 0
        total_extra = GROW * max(len(msgs) - 1, 0)
        if self.path.split("?")[0].endswith("/messages"):
            usage = {"input_tokens": 20, "output_tokens": 1,
                     "cache_read_input_tokens": (big + extra) if hit else 0,
                     "cache_creation_input_tokens": (total_extra - extra) if hit else big + total_extra}
            self._sse([
                {"type": "message_start", "message": {"usage": usage}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "你好"}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "世界"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}},
                {"type": "message_stop"},
            ])
        else:
            usage = {"input_tokens": big + 20 + total_extra, "output_tokens": 8,
                     "input_tokens_details": {"cached_tokens": (big + extra) if hit else 0},
                     "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": big + 28 + total_extra}
            self._sse([
                {"type": "response.created", "response": {}},
                {"type": "response.output_text.delta", "delta": "你好"},
                {"type": "response.output_text.delta", "delta": "世界"},
                {"type": "response.completed", "response": {"usage": usage}},
            ])


def serve(port=0):
    ThreadingHTTPServer.request_queue_size = 128
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    MODE.update(delay=0.03, ttft=(0.25, 0.7))
    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 18080)
    print("mock on", s.server_address)
    threading.Event().wait()
