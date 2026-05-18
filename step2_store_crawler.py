"""
crawling/step2_store_crawler.py  ← 이승현 담당
────────────────────────────────────────────────────────────────
2단계: url 테이블의 naver_url 기반으로
       업체 상세정보 + 리뷰 크롤링
→ raw_store, raw_review 테이블에 저장

실행 예)
  python step2_store_crawler.py --batch 50
────────────────────────────────────────────────────────────────
"""
import re
import time
import json
import argparse
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from utils.db import get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://map.naver.com/",
}

# 네이버 플레이스 API 엔드포인트
PLACE_API   = "https://place.map.naver.com/place/main/summary/{code}"
REVIEW_API  = "https://place.map.naver.com/place/main/review/{code}"
MENU_API    = "https://place.map.naver.com/place/main/menu/{code}"
INFO_API    = "https://place.map.naver.com/place/main/info/{code}"


def fetch_json(url: str, session: requests.Session) -> dict:
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("fetch 실패 url=%s err=%s", url, e)
        return {}


def parse_store(code: str, session: requests.Session) -> dict:
    """업체 홈 탭 — 기본 정보"""
    data  = fetch_json(PLACE_API.format(code=code), session)
    info  = data.get("summary", {})
    return {
        "store_name":     info.get("name", ""),
        "address":        info.get("address", ""),
        "phone":          info.get("phone", ""),
        "business_hours": json.dumps(info.get("businessHours", {}),
                                     ensure_ascii=False),
        "store_image_url": (info.get("images") or [{}])[0].get("url", ""),
        "description":    info.get("description", ""),
        "category_raw":   info.get("category", ""),
    }


def parse_menu(code: str, session: requests.Session) -> tuple[str, str]:
    """메뉴 탭"""
    data  = fetch_json(MENU_API.format(code=code), session)
    menu_url = f"https://place.map.naver.com/place/main/menu/{code}"
    menus = data.get("menus", [])
    menu_text = " | ".join(
        f"{m.get('name','')}({m.get('price','')})" for m in menus[:30]
    )
    return menu_text, menu_url


def parse_convenience(code: str, session: requests.Session) -> list[str]:
    """정보 탭 — 편의시설 및 서비스"""
    data = fetch_json(INFO_API.format(code=code), session)
    return data.get("conveniences", [])


def parse_reviews(
    code: str, url_id: int, session: requests.Session
) -> list[dict]:
    """리뷰 탭 — 방문자 + 블로그 리뷰"""
    reviews: list[dict] = []
    data = fetch_json(REVIEW_API.format(code=code), session)

    # 방문자 리뷰
    for r in data.get("visitorReviews", []):
        reviews.append({
            "url_id":      url_id,
            "store_code":  code,
            "content":     r.get("body", ""),
            "written_dt":  r.get("visitDate", None),
            "review_type": "V",
            "status":      "N",
        })

    # 블로그 리뷰
    for r in data.get("blogReviews", []):
        reviews.append({
            "url_id":      url_id,
            "store_code":  code,
            "content":     r.get("description", ""),
            "written_dt":  r.get("postDate", None),
            "review_type": "B",
            "status":      "N",
        })

    return reviews


def get_pending_urls(batch: int) -> list[dict]:
    """is_done='N' 인 url 목록 가져오기"""
    sql = """
        SELECT url_id, store_code, naver_url
        FROM url
        WHERE is_done = 'N'
        LIMIT %(batch)s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"batch": batch})
            return cur.fetchall()


def save_raw_store(store: dict):
    sql = """
        INSERT IGNORE INTO raw_store
          (url_id, store_code, store_name, address, phone,
           business_hours, store_image_url, menu_url, description,
           menu, category_raw, review_count, is_processed, created_at)
        VALUES
          (%(url_id)s, %(store_code)s, %(store_name)s, %(address)s, %(phone)s,
           %(business_hours)s, %(store_image_url)s, %(menu_url)s, %(description)s,
           %(menu)s, %(category_raw)s, %(review_count)s, 'N', NOW())
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, store)


def save_raw_reviews(reviews: list[dict]):
    if not reviews:
        return
    sql = """
        INSERT INTO raw_review
          (url_id, store_code, content, written_dt,
           review_type, status, created_at)
        VALUES
          (%(url_id)s, %(store_code)s, %(content)s, %(written_dt)s,
           %(review_type)s, 'N', NOW())
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, reviews)


def mark_done(url_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE url SET is_done='Y' WHERE url_id=%s", (url_id,)
            )


def crawl_one(row: dict, session: requests.Session):
    code   = row["store_code"]
    url_id = row["url_id"]
    log.info("크롤링 시작: %s", code)

    store_info   = parse_store(code, session)
    menu, menu_url = parse_menu(code, session)
    reviews      = parse_reviews(code, url_id, session)

    store_row = {
        "url_id":    url_id,
        "store_code": code,
        **store_info,
        "menu":         menu,
        "menu_url":     menu_url,
        "review_count": len(reviews),
    }

    save_raw_store(store_row)
    save_raw_reviews(reviews)
    mark_done(url_id)

    log.info("완료: %s | 리뷰 %d건", code, len(reviews))
    time.sleep(2)   # 서버 부하 방지


def main():
    parser = argparse.ArgumentParser(description="업체 상세정보 + 리뷰 크롤러")
    parser.add_argument("--batch", type=int, default=50,
                        help="한 번에 처리할 업체 수")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    pending = get_pending_urls(args.batch)
    log.info("처리 대상: %d건", len(pending))

    for row in pending:
        try:
            crawl_one(row, session)
        except Exception as e:
            log.error("오류 store_code=%s: %s", row["store_code"], e)


if __name__ == "__main__":
    main()
