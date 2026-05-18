"""
run_pipeline.py
────────────────────────────────────────────────────────────────
전체 파이프라인 순서 실행 + 스케줄러 (cron 대체)

  단발 실행:
    python run_pipeline.py --once

  스케줄 실행 (매일 새벽 3시):
    python run_pipeline.py --schedule

  특정 단계만:
    python run_pipeline.py --steps crawl preprocess
────────────────────────────────────────────────────────────────
"""
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

PYTHON = sys.executable

STEP_COMMANDS: list[tuple[str, list[str]]] = [
    ("1. URL 수집",       [PYTHON, "crawling/step1_url_collector.py",
                           "--area", "A", "--area", "B", "--area", "C",
                           "--max", "300"]),
    ("2. 상세정보·리뷰",  [PYTHON, "crawling/step2_store_crawler.py",
                           "--batch", "100"]),
    ("3. 전처리",         [PYTHON, "preprocessing/step3_preprocess.py"]),
    ("4. 감성분석",       [PYTHON, "sentiment/step4_sentiment.py",
                           "--mode", "infer"]),
    ("5. 점수화",         [PYTHON, "scoring/step5_scoring.py"]),
]

STEP_ALIAS = {
    "crawl":      ["1. URL 수집", "2. 상세정보·리뷰"],
    "preprocess": ["3. 전처리"],
    "sentiment":  ["4. 감성분석"],
    "score":      ["5. 점수화"],
    "all":        None,   # 전체
}


def run_command(name: str, cmd: list[str]) -> bool:
    log.info("▶ 시작: %s", name)
    start = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - start).seconds
    if result.returncode == 0:
        log.info("✓ 완료: %s (%ds)", name, elapsed)
        return True
    else:
        log.error("✗ 실패: %s (returncode=%d)", name, result.returncode)
        return False


def run_pipeline(steps_filter: list[str] | None = None):
    log.info("=" * 50)
    log.info("파이프라인 시작: %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 50)

    success_all = True
    for name, cmd in STEP_COMMANDS:
        if steps_filter and name not in steps_filter:
            continue
        ok = run_command(name, cmd)
        if not ok:
            success_all = False
            log.error("파이프라인 중단 — %s 실패", name)
            break

    status = "성공" if success_all else "실패"
    log.info("=" * 50)
    log.info("파이프라인 %s: %s", status,
             datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="맛집 데이터 파이프라인")
    parser.add_argument("--once",     action="store_true",
                        help="즉시 1회 실행")
    parser.add_argument("--schedule", action="store_true",
                        help="매일 03:00 자동 실행")
    parser.add_argument("--steps",    nargs="+",
                        choices=list(STEP_ALIAS.keys()), default=["all"])
    args = parser.parse_args()

    # 실행할 단계 이름 목록
    target_names: list[str] | None = None
    if "all" not in args.steps:
        target_names = []
        for alias in args.steps:
            names = STEP_ALIAS.get(alias)
            if names:
                target_names.extend(names)

    if args.once or not args.schedule:
        run_pipeline(target_names)

    if args.schedule:
        import schedule
        import time
        log.info("스케줄러 등록: 매일 03:00")
        schedule.every().day.at("03:00").do(run_pipeline, target_names)
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    main()
