"""
crawling/step1_url_collector.py  ← 조태현 담당
────────────────────────────────────────────────────────────────
1단계: 네이버 지도에서 지역별 업체 URL / store_code 수집
→ url 테이블에 INSERT

지역 코드 (area)
  A : 전북대  B : 신시가지  C : 객사(한옥마을 포함)

실행 예)
  python step1_url_collector.py --area A --keyword 음식점 --max 200
────────────────────────────────────────────────────────────────
"""
import re
import time
import argparse
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from utils.db import get_conn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 지역별 네이버 지도 검색 기준 좌표 ─────────────────────────────
AREA_COORDS = {
    "A": {"name": "전북대",    "lat": 35.8468, "lng": 127.1326},
    "B": {"name": "신시가지",  "lat": 35.8328, "lng": 127.1087},
    "C": {"name": "객사",      "lat": 35.8191, "lng": 127.1494},
}

NAVER_SEARCH_URL = "https://map.naver.com/v5/api/search"
STORE_DETAIL_BASE = "https://map.naver.com/v5/entry/place/{store_code}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://map.naver.com/",
}


def extract_store_code(place_url: str) -> str | None:
    """네이버 place URL에서 업체 일련번호 추출"""
    m = re.search(r"/place/(\d+)", place_url)
    return m.group(1) if m else None


def search_places(area_code: str, keyword: str, max_count: int = 200) -> list[dict]:
    """
    네이버 지도 검색 API로 업체 목록 수집
    반환: [{"store_code": "...", "naver_url": "..."}, ...]
    """
    coord = AREA_COORDS[area_code]
    results: list[dict] = []
    page = 1

    session = requests.Session()
    session.headers.update(HEADERS)

    while len(results) < max_count:
        params = {
            "query":   f"{coord['name']} {keyword}",
            "type":    "place",
            "page":    page,
            "display": 40,
            "lang":    "ko",
        }
        try:
            resp = session.get(NAVER_SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("검색 API 오류: %s", e)
            break

        places = data.get("result", {}).get("place", {}).get("list", [])
        if not places:
            break

        for p in places:
            store_code = p.get("id", "")
            if not store_code:
                continue
            naver_url = STORE_DETAIL_BASE.format(store_code=store_code)
            results.append({
                "store_code": store_code,
                "naver_url":  naver_url,
                "area":       area_code,
            })
            if len(results) >= max_count:
                break

        log.info("page=%d  누적=%d", page, len(results))
        page += 1
        time.sleep(1.5)   # 서버 부하 방지

    return results


def save_urls(records: list[dict]) -> int:
    """url 테이블에 UPSERT (이미 있는 store_code 는 스킵)"""
    if not records:
        return 0

    sql = """
        INSERT IGNORE INTO url (store_code, naver_url, area, is_done, created_at)
        VALUES (%(store_code)s, %(naver_url)s, %(area)s, 'N', NOW())
    """
    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in records:
                cur.execute(sql, r)
                inserted += cur.rowcount
    return inserted


def main():
    parser = argparse.ArgumentParser(description="네이버 지도 업체 URL 수집기")
    parser.add_argument("--area",    required=True, choices=["A", "B", "C"],
                        help="A:전북대  B:신시가지  C:객사")
    parser.add_argument("--keyword", default="음식점")
    parser.add_argument("--max",     type=int, default=200)
    args = parser.parse_args()

    log.info("=== URL 수집 시작 | area=%s keyword=%s max=%d ===",
             args.area, args.keyword, args.max)

    records = search_places(args.area, args.keyword, args.max)
    inserted = save_urls(records)

    log.info("=== 완료 | 수집=%d  신규 저장=%d ===", len(records), inserted)


if __name__ == "__main__":
    main()
