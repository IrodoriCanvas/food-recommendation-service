"""
scoring/step5_scoring.py  ← 전원 (회의 결론 점수화 공식 구현)
────────────────────────────────────────────────────────────────
5단계: 100점 만점 점수화
  맛 점수     50점  — 긍정 리뷰 비율 (맛 관련 키워드 포함 리뷰 / 전체)
  최근성 점수 20점  — 최근 3개월 리뷰 / 전체 리뷰 비율
  편의성 점수 15점  — 편의시설 항목 수 (5개 이상 만점)
  계절적합성  15점  — Sseason 공식 (시간비중·메뉴매칭·텍스트언급)

→ review_result, analyze_report 저장

실행 예)
  python step5_scoring.py --season 봄
────────────────────────────────────────────────────────────────
"""
import argparse
import logging
import re
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import numpy as np

from utils.db import get_conn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 계절 정의 ─────────────────────────────────────────────────────
SEASONS: dict[str, tuple[int, ...]] = {
    "봄":  (3, 4, 5),
    "여름": (6, 7, 8),
    "가을": (9, 10, 11),
    "겨울": (12, 1, 2),
}

SEASON_KEYWORDS: dict[str, list[str]] = {
    "봄":  ["봄", "벚꽃", "나물", "봄나물", "달래", "냉이"],
    "여름": ["여름", "냉면", "빙수", "시원", "냉국", "장어"],
    "가을": ["가을", "버섯", "전어", "대하", "햅쌀", "단풍"],
    "겨울": ["겨울", "뜨끈", "어묵", "굴", "동태", "꼬막", "전골"],
}

MENU_SEASON_KEYWORDS: dict[str, list[str]] = {
    "봄":  ["봄나물", "봄특선", "벚꽃"],
    "여름": ["냉면", "빙수", "삼계탕", "냉국수"],
    "가을": ["전어구이", "버섯전골", "대하구이"],
    "겨울": ["굴전골", "동태찌개", "꼬막비빔밥"],
}

TASTE_KEYWORDS = [
    "맛있", "맛나", "맛짱", "존맛", "JMT", "맛 좋", "대박",
    "신선", "재료", "맛집", "맛보", "환상적",
]

FACILITY_ITEMS = [
    "주차", "단체", "포장", "배달", "예약", "유아", "아동",
    "무선인터넷", "wifi", "wi-fi",
]


# ──────────────────────────────────────────────────────────────────
# 점수 계산 함수
# ──────────────────────────────────────────────────────────────────

def score_taste(reviews: pd.DataFrame) -> tuple[int, str]:
    """
    맛 점수 (50점 만점)
    공식: (맛 키워드 포함 긍정 리뷰 수 / 전체 리뷰 수) * 50
    """
    if reviews.empty:
        return 0, ""

    def has_taste_kw(text: str) -> bool:
        return any(kw in text for kw in TASTE_KEYWORDS)

    mask_pos   = reviews["sentiment"] == "P"
    mask_taste = reviews["content"].apply(has_taste_kw)

    taste_positive = (mask_pos & mask_taste).sum()
    total          = len(reviews)

    ratio = taste_positive / total if total > 0 else 0
    score = min(int(ratio * 50), 50)

    # 점수화에 사용된 대표 문장 (최대 5개)
    sample = reviews[mask_pos & mask_taste]["content"].head(5).tolist()
    content_str = " | ".join(sample[:3])

    return score, content_str


def score_recency(reviews: pd.DataFrame) -> tuple[int, str]:
    """
    최근성 점수 (20점 만점)
    공식: 최근 3개월 리뷰 수 / 전체 리뷰 수 * 20
    """
    if reviews.empty:
        return 0, ""

    cutoff = date.today() - timedelta(days=90)
    recent = reviews[pd.to_datetime(reviews["written_dt"]).dt.date >= cutoff]

    ratio = len(recent) / len(reviews) if len(reviews) > 0 else 0
    score = min(int(ratio * 20), 20)

    sample = recent["content"].head(3).tolist()
    content_str = " | ".join(sample)

    return score, content_str


def score_facility(description: str, menu: str) -> tuple[int, str]:
    """
    편의성 점수 (15점 만점)
    공식: 편의시설 항목 5개 이상 만점, 그 이하 3점씩 차감
    회의 결론: 편의성 항목이 없는 경우 0점 시작
    """
    text = f"{description or ''} {menu or ''}".lower()
    found = [item for item in FACILITY_ITEMS if item.lower() in text]
    count = len(found)

    if count >= 5:
        score = 15
    else:
        score = max(count * 3, 0)

    return score, ", ".join(found)


def score_season(
    reviews: pd.DataFrame, menu: str, season: str
) -> tuple[int, str]:
    """
    계절 적합성 점수 (15점 만점)
    공식: Sseason = (A*0.5) + (B*0.3) + (C*0.2)
      A: 해당 계절 리뷰 수 / 연간 총 리뷰 수
      B: 메뉴에 계절 키워드 포함 여부 (0 or 1)
      C: 리뷰 본문 계절 언급 비율
    결과: Sseason * 15
    """
    if reviews.empty or season not in SEASONS:
        return 0, ""

    target_months = SEASONS[season]
    text_kws      = SEASON_KEYWORDS.get(season, [])
    menu_kws      = MENU_SEASON_KEYWORDS.get(season, [])

    reviews = reviews.copy()
    reviews["month"] = pd.to_datetime(reviews["written_dt"]).dt.month

    # A: 시간 비중
    season_reviews = reviews[reviews["month"].isin(target_months)]
    A = len(season_reviews) / len(reviews) if len(reviews) > 0 else 0

    # B: 메뉴 매칭
    menu_text = (menu or "").lower()
    B = 1.0 if any(kw.lower() in menu_text for kw in menu_kws) else 0.0

    # C: 텍스트 언급 비율
    def has_season_kw(text: str) -> bool:
        return any(kw in text for kw in text_kws)
    C = reviews["content"].apply(has_season_kw).mean()

    S_season = (A * 0.5) + (B * 0.3) + (C * 0.2)
    score = min(int(S_season * 15), 15)

    matched = season_reviews["content"].head(3).tolist()
    content_str = " | ".join(matched)

    return score, content_str


# ──────────────────────────────────────────────────────────────────
# 메인 로직
# ──────────────────────────────────────────────────────────────────

def load_stores() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT final_store_id, store_code, menu, description FROM final_store",
            conn,
        )


def load_reviews(store_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT content, written_dt, review_type, sentiment "
            "FROM final_review WHERE final_store_id = %s",
            conn,
            params=(store_id,),
        )


def save_result(row: dict):
    sql = """
        INSERT INTO review_result
          (final_store_id, taste_content, recent_content,
           facility_content, season_content,
           taste_score, recent_score, facility_score,
           season_score, total_score)
        VALUES
          (%(final_store_id)s, %(taste_content)s, %(recent_content)s,
           %(facility_content)s, %(season_content)s,
           %(taste_score)s, %(recent_score)s, %(facility_score)s,
           %(season_score)s, %(total_score)s)
        ON DUPLICATE KEY UPDATE
          taste_score=VALUES(taste_score),
          recent_score=VALUES(recent_score),
          facility_score=VALUES(facility_score),
          season_score=VALUES(season_score),
          total_score=VALUES(total_score)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, row)


def save_report(store_id: int, total_score: int, review_sum: str):
    sql = """
        INSERT INTO analyze_report
          (final_store_id, review_sum, total_score, result_at)
        VALUES (%s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
          total_score=VALUES(total_score),
          review_sum=VALUES(review_sum),
          result_at=NOW()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (store_id, review_sum, total_score))


def process_store(store: dict, season: str):
    sid   = store["final_store_id"]
    menu  = store.get("menu", "") or ""
    desc  = store.get("description", "") or ""

    reviews = load_reviews(sid)

    t_score, t_content = score_taste(reviews)
    r_score, r_content = score_recency(reviews)
    f_score, f_content = score_facility(desc, menu)
    s_score, s_content = score_season(reviews, menu, season)

    total = t_score + r_score + f_score + s_score

    save_result({
        "final_store_id":   sid,
        "taste_content":    t_content,
        "recent_content":   r_content,
        "facility_content": f_content,
        "season_content":   s_content,
        "taste_score":      t_score,
        "recent_score":     r_score,
        "facility_score":   f_score,
        "season_score":     s_score,
        "total_score":      total,
    })

    # analyze_report 에는 대표 리뷰 요약 저장 (추후 BERT 요약으로 대체 가능)
    review_sum = t_content or r_content or ""
    save_report(sid, total, review_sum)

    log.info(
        "store_id=%s | 맛=%d 최근=%d 편의=%d 계절=%d → 총점=%d",
        sid, t_score, r_score, f_score, s_score, total,
    )


def main():
    parser = argparse.ArgumentParser(description="맛집 점수화")
    parser.add_argument(
        "--season", choices=["봄", "여름", "가을", "겨울"],
        default=_current_season(),
        help="계절적합성 계산 기준 계절 (기본: 현재 계절)"
    )
    args = parser.parse_args()

    log.info("=== 점수화 시작 | 기준 계절: %s ===", args.season)
    stores = load_stores()
    log.info("대상 업체: %d건", len(stores))

    for _, store in stores.iterrows():
        try:
            process_store(store.to_dict(), args.season)
        except Exception as e:
            log.error("오류 store_id=%s: %s", store["final_store_id"], e)

    log.info("=== 점수화 완료 ===")


def _current_season() -> str:
    m = date.today().month
    if m in (3, 4, 5):   return "봄"
    if m in (6, 7, 8):   return "여름"
    if m in (9, 10, 11): return "가을"
    return "겨울"


if __name__ == "__main__":
    main()
