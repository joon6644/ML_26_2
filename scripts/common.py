"""수집 스크립트 공통 유틸: .env 로드, 재시도 HTTP, 경로, 수집 기간."""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

START = "2016-01-01"   # 학습 기간 시작 (docs/data_inventory.md 참고)
END = "2026-09-30"


def load_env():
    env = {}
    for line in open(ROOT / ".env", encoding="utf-8"):
        m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
        if m:
            env[m[1]] = m[2].strip().strip('"').strip("'")
    return env


ENV = load_env()


class QuotaExceeded(Exception):
    pass


# 하루 호출 한도 초과 응답 (data.go.kr 게이트웨이는 영문, 축평원 등 기관 서버는 한글로 준다)
QUOTA_MARKERS = ("LIMITED_NUMBER_OF_SERVICE_REQUESTS", "요청제한 횟수 초과")


def http_get(url, params=None, safe="", retries=4, timeout=90):
    """GET 후 본문 문자열 반환. data.go.kr cond[...] 파라미터는 safe='[]:'로 대괄호를 그대로 보낸다."""
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, safe=safe)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")
            if any(k in body for k in QUOTA_MARKERS):
                raise QuotaExceeded(url.split("?")[0])
            return body
        except QuotaExceeded:
            raise
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if any(k in body for k in QUOTA_MARKERS):
                raise QuotaExceeded(url.split("?")[0])
            last = f"HTTP {e.code}: {body[:200]}"
            if e.code < 500 and e.code != 429:
                break
        except Exception as e:  # 네트워크 오류는 재시도
            last = repr(e)
        time.sleep(2 ** i)
    raise RuntimeError(f"요청 실패: {url.split('?')[0]} ({last})")


def http_json(url, params=None, **kw):
    return json.loads(http_get(url, params, **kw))
