"""
utils/db.py
DB 연결 및 공통 유틸리티
테이블정의서 기준 컬럼명 그대로 사용
"""
import os
import pymysql
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "db":       os.getenv("DB_NAME", "matzip"),
    "charset":  "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": False,
}


@contextmanager
def get_conn():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_tables():
    """
    테이블정의서(sms) 기준으로 테이블 생성
    이미 존재하면 스킵
    """
    ddl_statements = [
        # ── url ────────────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS url (
            url_id      INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
            store_code  VARCHAR(30)  NOT NULL UNIQUE COMMENT '업체 일련번호',
            naver_url   VARCHAR(255) COMMENT '업체 상세정보 url',
            menu_code   VARCHAR(30)  UNIQUE,
            area        CHAR(1)      COMMENT 'A:전북대 B:신시가지 C:객사',
            is_done     CHAR(1)      DEFAULT 'N' COMMENT 'Y:완료 N:미완',
            created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── raw_store ───────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS raw_store (
            raw_store_id   INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
            url_id         INT           NOT NULL,
            store_code     VARCHAR(30)   NOT NULL UNIQUE,
            store_name     VARCHAR(255),
            address        VARCHAR(255),
            phone          VARCHAR(20),
            business_hours VARCHAR(1000),
            store_image_url VARCHAR(255),
            menu_url       VARCHAR(255),
            description    VARCHAR(2000),
            menu           VARCHAR(6000),
            category_raw   VARCHAR(255)  COMMENT '원본 카테고리 (순댓국, 피자 등)',
            review_count   INT           DEFAULT 0,
            is_processed   CHAR(1)       DEFAULT 'N',
            created_at     DATETIME      DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES url(url_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── final_store ─────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS final_store (
            final_store_id  INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
            url_id          INT           NOT NULL,
            store_code      VARCHAR(30)   NOT NULL UNIQUE,
            store_name      VARCHAR(255),
            address         VARCHAR(255),
            phone           VARCHAR(20),
            business_hours  VARCHAR(1000),
            is_active       CHAR(1)       DEFAULT 'Y' COMMENT 'Y:영업 N:폐업',
            store_image_url VARCHAR(255),
            menu_url        VARCHAR(255),
            description     VARCHAR(2000),
            menu            VARCHAR(6000),
            category_raw    VARCHAR(255),
            category_mapped VARCHAR(50)   COMMENT '한식/중식/일식/양식/카페/기타',
            review_count    INT           DEFAULT 0,
            created_at      DATETIME      DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES url(url_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── keyword_list ────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS keyword_list (
            keyword_id INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
            type       CHAR(1)      COMMENT 's:계절 r:지역 f:음식종류',
            content    VARCHAR(10)  COMMENT '봄/여름/가을/겨울 | 전북대/객사/신시가지 | 한식/중식...'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── store_keyword ───────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS store_keyword (
            st_store_id   INT NOT NULL,
            st_keyword_id INT NOT NULL,
            PRIMARY KEY (st_store_id, st_keyword_id),
            FOREIGN KEY (st_store_id)   REFERENCES final_store(final_store_id),
            FOREIGN KEY (st_keyword_id) REFERENCES keyword_list(keyword_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── raw_review ──────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS raw_review (
            raw_review_id INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
            url_id        INT           NOT NULL,
            store_code    VARCHAR(30),
            content       VARCHAR(6000),
            written_dt    DATE,
            review_type   CHAR(1)       NOT NULL COMMENT 'V:방문자 B:블로그',
            status        CHAR(1)       DEFAULT 'N' COMMENT 'Y:처리완료',
            created_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES url(url_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── final_review ────────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS final_review (
            final_review_id INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
            final_store_id  INT           NOT NULL,
            store_code      VARCHAR(30),
            content         VARCHAR(6000),
            written_dt      DATE,
            review_type     CHAR(1)       NOT NULL COMMENT 'V:방문자 B:블로그',
            sentiment       CHAR(1)       COMMENT 'P:긍정 N:부정 U:중립',
            FOREIGN KEY (final_store_id) REFERENCES final_store(final_store_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── review_result ───────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS review_result (
            result_id        INT  NOT NULL AUTO_INCREMENT PRIMARY KEY,
            final_store_id   INT  NOT NULL,
            taste_content    VARCHAR(3000) COMMENT '맛 관련 문장 모음',
            recent_content   VARCHAR(3000) COMMENT '최근 3개월 리뷰 문장',
            facility_content VARCHAR(3000) COMMENT '편의성 문장',
            season_content   VARCHAR(3000) COMMENT '계절 관련 문장',
            taste_score      INT  DEFAULT 0 COMMENT '맛 점수 /50',
            recent_score     INT  DEFAULT 0 COMMENT '최근성 /20',
            facility_score   INT  DEFAULT 0 COMMENT '편의성 /15',
            season_score     INT  DEFAULT 0 COMMENT '계절적합성 /15',
            total_score      INT  DEFAULT 0 COMMENT '총점 /100',
            FOREIGN KEY (final_store_id) REFERENCES final_store(final_store_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,

        # ── analyze_report ──────────────────────────────────────────────
        """
        CREATE TABLE IF NOT EXISTS analyze_report (
            report_id      INT           NOT NULL AUTO_INCREMENT PRIMARY KEY,
            final_store_id INT           NOT NULL,
            review_sum     VARCHAR(3000) COMMENT 'KcBERT/BERT 리뷰 요약',
            total_score    INT           DEFAULT 0,
            result_at      DATETIME      DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (final_store_id) REFERENCES final_store(final_store_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            for ddl in ddl_statements:
                cur.execute(ddl)
    print("[init_tables] 모든 테이블 생성/확인 완료")


if __name__ == "__main__":
    init_tables()
