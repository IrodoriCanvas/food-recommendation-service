"""
preprocessing/step3_preprocess.py  ← 송민석 + 손민건 담당
────────────────────────────────────────────────────────────────
3단계: raw_store / raw_review 전처리
  1) 결측치 제거 (stars=None, text='')
  2) 중복 제거 (store_code + content 기준)
  3) 카테고리 리매핑 (category_raw → 한식/중식/일식/양식/카페/기타)
  4) final_store / final_review 저장

실행 예)
  python step3_preprocess.py
────────────────────────────────────────────────────────────────
"""
import re
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from utils.db import get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 카테고리 매핑 사전 (회의 결론 기반) ─────────────────────────────
# 1차: 명확한 키워드
CATEGORY_MAP_PRIMARY: dict[str, list[str]] = {
    "한식": ["한식", "국밥", "삼겹살", "갈비", "불고기", "비빔밥", "순두부",
             "삼계탕", "된장", "김치찌개", "순댓국", "설렁탕", "냉면",
             "해장국", "전골", "구이", "백반", "찌개", "보쌈"],
    "중식": ["중식", "짜장", "짬뽕", "탕수육", "마파두부", "딤섬", "중국"],
    "일식": ["일식", "라멘", "스시", "초밥", "돈카츠", "우동", "소바",
             "이자카야", "야키토리", "규동", "텐동", "일본"],
    "양식": ["양식", "파스타", "스테이크", "피자", "리조또", "버거",
             "샐러드", "브런치", "이탈리안", "프렌치"],
    "카페": ["카페", "커피", "디저트", "베이커리", "브런치카페", "tea"],
}

# 2차: 네이버 크롤링 raw 카테고리에 자주 나오는 세부 표현
CATEGORY_MAP_EXTENDED: dict[str, list[str]] = {
    "한식": ["고기", "구이집", "포차", "어묵", "떡볶이", "순대", "분식"],
    "중식": ["중국집"],
    "일식": ["라멘집", "스시집"],
    "양식": ["샌드위치", "수제버거", "그릴"],
    "카페": ["카페베이커리", "로스터리", "핸드드립"],
}


def detect_category(raw: str) -> str:
    """
    category_raw 텍스트에서 매핑된 카테고리 반환
    → 1차 키워드 → 2차 키워드 → '기타'
    """
    if not raw:
        return "기타"
    text = raw.lower()

    for cat, keywords in CATEGORY_MAP_PRIMARY.items():
        if any(kw.lower() in text for kw in keywords):
            return cat

    for cat, keywords in CATEGORY_MAP_EXTENDED.items():
        if any(kw.lower() in text for kw in keywords):
            return cat

    return "기타"


# ── DB에서 raw 데이터 로드 ────────────────────────────────────────
def load_raw_stores() -> pd.DataFrame:
    sql = """
        SELECT raw_store_id, url_id, store_code, store_name,
               address, phone, business_hours, store_image_url,
               menu_url, description, menu, category_raw, review_count
        FROM raw_store
        WHERE is_processed = 'N'
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn)


def load_raw_reviews() -> pd.DataFrame:
    sql = """
        SELECT raw_review_id, url_id, store_code, content,
               written_dt, review_type
        FROM raw_review
        WHERE status = 'N'
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn)


# ── 전처리 ────────────────────────────────────────────────────────
def preprocess_stores(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)

    # 1) 필수 컬럼 결측치 제거
    df["store_name"] = df["store_name"].replace("", np.nan)
    df = df.dropna(subset=["store_code", "store_name"])

    # 2) 중복 제거 (store_code 기준)
    df = df.drop_duplicates(subset=["store_code"])

    # 3) 카테고리 리매핑
    df["category_mapped"] = df["category_raw"].apply(detect_category)

    # 4) url_id 연결 확인 (null 제거)
    df = df.dropna(subset=["url_id"])
    df["url_id"] = df["url_id"].astype(int)

    log.info("업체 전처리: %d → %d 건", original, len(df))
    return df.reset_index(drop=True)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)

    # 1) 본문 결측치 제거 (None, 빈 문자열)
    df["content"] = df["content"].replace("", np.nan)
    df = df.dropna(subset=["content"])

    # 2) 중복 제거 (store_code + content 기준)
    df = df.drop_duplicates(subset=["store_code", "content"])

    # 3) 날짜 표준화 (25.02.10 → 2025-02-10)
    df["written_dt"] = pd.to_datetime(
        df["written_dt"], errors="coerce"
    ).dt.date

    # 4) 날짜 없는 리뷰 제거
    df = df.dropna(subset=["written_dt"])

    log.info("리뷰 전처리: %d → %d 건", original, len(df))
    return df.reset_index(drop=True)


# ── DB 저장 ───────────────────────────────────────────────────────
def save_final_stores(df: pd.DataFrame):
    sql = """
        INSERT IGNORE INTO final_store
          (url_id, store_code, store_name, address, phone,
           business_hours, store_image_url, menu_url, description,
           menu, category_raw, category_mapped, review_count, created_at)
        VALUES
          (%(url_id)s, %(store_code)s, %(store_name)s, %(address)s, %(phone)s,
           %(business_hours)s, %(store_image_url)s, %(menu_url)s, %(description)s,
           %(menu)s, %(category_raw)s, %(category_mapped)s, %(review_count)s, NOW())
    """
    records = df[[
        "url_id", "store_code", "store_name", "address", "phone",
        "business_hours", "store_image_url", "menu_url", "description",
        "menu", "category_raw", "category_mapped", "review_count",
    ]].to_dict("records")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, records)
            # raw_store is_processed 플래그 업데이트
            codes = tuple(df["store_code"].tolist())
            if codes:
                cur.execute(
                    "UPDATE raw_store SET is_processed='Y' "
                    f"WHERE store_code IN {codes if len(codes)>1 else f'(\"{codes[0]}\")' }"
                )
    log.info("final_store 저장: %d건", len(records))


def save_final_reviews(df: pd.DataFrame):
    """
    final_store 의 final_store_id 를 JOIN 하여 final_review 에 저장
    """
    sql_get_ids = "SELECT final_store_id, store_code FROM final_store"
    with get_conn() as conn:
        id_df = pd.read_sql(sql_get_ids, conn)

    merged = df.merge(id_df, on="store_code", how="inner")
    if merged.empty:
        log.warning("final_review 저장 대상 없음 (store_code 매핑 실패)")
        return

    sql = """
        INSERT INTO final_review
          (final_store_id, store_code, content, written_dt, review_type)
        VALUES
          (%(final_store_id)s, %(store_code)s, %(content)s,
           %(written_dt)s, %(review_type)s)
    """
    records = merged[[
        "final_store_id", "store_code", "content", "written_dt", "review_type"
    ]].to_dict("records")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, records)
            # raw_review status 업데이트
            raw_ids = tuple(df["raw_review_id"].tolist())
            if raw_ids:
                cur.execute(
                    f"UPDATE raw_review SET status='Y' WHERE raw_review_id IN "
                    f"{raw_ids if len(raw_ids)>1 else f'({raw_ids[0]})'}"
                )
    log.info("final_review 저장: %d건", len(records))


# ── 엔트리포인트 ──────────────────────────────────────────────────
def main():
    log.info("=== 전처리 시작 ===")

    # 업체 전처리
    raw_stores = load_raw_stores()
    if raw_stores.empty:
        log.info("처리할 raw_store 없음")
    else:
        final_stores = preprocess_stores(raw_stores)
        save_final_stores(final_stores)

    # 리뷰 전처리
    raw_reviews = load_raw_reviews()
    if raw_reviews.empty:
        log.info("처리할 raw_review 없음")
    else:
        final_reviews = preprocess_reviews(raw_reviews)
        save_final_reviews(final_reviews)

    log.info("=== 전처리 완료 ===")


if __name__ == "__main__":
    main()
