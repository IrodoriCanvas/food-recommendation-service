"""
sentiment/step4_sentiment.py  ← 신선양 담당
────────────────────────────────────────────────────────────────
4단계: KcBERT 파인튜닝 + 감성분석 (긍/부정/중립)
  회의 결론: 방법 1 — 2차 학습
    1차: NSMC 20만건 (영화 리뷰, 긍·부정 학습)
    2차: 크롤링한 맛집 리뷰 ~1000건 (맛집 특화 표현 학습)

  실행 모드
    --mode train   : KcBERT 파인튜닝 실행
    --mode infer   : final_review.sentiment 업데이트
    --mode both    : 학습 후 추론까지

  요구 패키지
    pip install transformers torch datasets scikit-learn
────────────────────────────────────────────────────────────────
"""
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────────
MODEL_NAME    = "snunlp/KR-FinBert-SC"   # KcBERT 계열 감성분석 특화 모델
# 또는 "beomi/kcbert-base" 로 변경 후 직접 파인튜닝 가능
SAVE_DIR      = Path("sentiment/model_ckpt")
MAX_LEN       = 128
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5

LABEL2ID = {"긍정": 0, "부정": 1, "중립": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SENTIMENT_CHAR = {"긍정": "P", "부정": "N", "중립": "U"}


# ──────────────────────────────────────────────────────────────────
# 1) 데이터 준비
# ──────────────────────────────────────────────────────────────────
def load_nsmc(path: str = "data/nsmc/ratings_train.txt") -> pd.DataFrame:
    """
    NSMC 데이터 로드 (https://github.com/e9t/nsmc)
    columns: id | document | label  (label 0=부정 1=긍정)
    """
    df = pd.read_csv(path, sep="\t").dropna()
    df["label_str"] = df["label"].map({1: "긍정", 0: "부정"})
    return df[["document", "label_str"]].rename(
        columns={"document": "text", "label_str": "label"}
    )


def load_matzip_reviews() -> pd.DataFrame:
    """
    final_review 에서 수동 레이블링된 리뷰 로드
    (테이블에 sentiment 컬럼이 이미 P/N/U 로 채워진 행)
    """
    from utils.db import get_conn
    sql = """
        SELECT content, sentiment
        FROM final_review
        WHERE sentiment IS NOT NULL
        LIMIT 2000
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    char2label = {"P": "긍정", "N": "부정", "U": "중립"}
    df["label"] = df["sentiment"].map(char2label)
    return df[["content", "label"]].rename(columns={"content": "text"})


def build_dataset(use_nsmc: bool = True, nsmc_sample: int = 5000):
    """
    학습용 데이터셋 구성
    회의 결론: 방법 1 (2차 학습) 또는 방법 2 (혼합)
    """
    from datasets import Dataset
    from sklearn.model_selection import train_test_split

    frames = []

    if use_nsmc and Path("data/nsmc/ratings_train.txt").exists():
        nsmc = load_nsmc()
        if nsmc_sample:
            nsmc = nsmc.sample(min(nsmc_sample, len(nsmc)), random_state=42)
        frames.append(nsmc)
        log.info("NSMC 데이터 %d건 로드", len(nsmc))

    try:
        matzip = load_matzip_reviews()
        frames.append(matzip)
        log.info("맛집 리뷰 레이블 데이터 %d건 로드", len(matzip))
    except Exception as e:
        log.warning("맛집 리뷰 로드 실패: %s", e)

    if not frames:
        raise RuntimeError("학습 데이터 없음 — NSMC 또는 레이블 리뷰 필요")

    combined = pd.concat(frames, ignore_index=True).dropna()
    combined["label_id"] = combined["label"].map(LABEL2ID)

    train_df, val_df = train_test_split(
        combined, test_size=0.1, random_state=42,
        stratify=combined["label_id"]
    )

    return (
        Dataset.from_pandas(train_df[["text", "label_id"]]),
        Dataset.from_pandas(val_df[["text", "label_id"]]),
    )


# ──────────────────────────────────────────────────────────────────
# 2) 학습
# ──────────────────────────────────────────────────────────────────
def train():
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding,
    )
    import evaluate

    log.info("=== KcBERT 파인튜닝 시작 ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds, val_ds = build_dataset()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    train_ds = train_ds.map(tokenize, batched=True).rename_column("label_id", "labels")
    val_ds   = val_ds.map(tokenize, batched=True).rename_column("label_id", "labels")

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    log.info("=== 학습 완료 → %s ===", SAVE_DIR)


# ──────────────────────────────────────────────────────────────────
# 3) 추론 + DB 업데이트
# ──────────────────────────────────────────────────────────────────
def infer():
    from transformers import pipeline
    from utils.db import get_conn

    model_path = str(SAVE_DIR) if SAVE_DIR.exists() else MODEL_NAME
    log.info("추론 모델 로드: %s", model_path)

    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1,          # CPU; GPU 사용시 device=0
        truncation=True,
        max_length=MAX_LEN,
    )

    # sentiment 가 아직 없는 리뷰만 처리
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT final_review_id, content FROM final_review "
            "WHERE sentiment IS NULL LIMIT 5000",
            conn,
        )

    if df.empty:
        log.info("추론 대상 없음")
        return

    log.info("추론 대상: %d건", len(df))

    texts = df["content"].tolist()
    preds = classifier(texts, batch_size=32)

    df["sentiment_label"] = [p["label"] for p in preds]
    df["sentiment_char"]  = df["sentiment_label"].map(
        lambda l: SENTIMENT_CHAR.get(l, "U")
    )

    # DB 업데이트
    sql = "UPDATE final_review SET sentiment=%s WHERE final_review_id=%s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            rows = list(zip(
                df["sentiment_char"].tolist(),
                df["final_review_id"].tolist(),
            ))
            cur.executemany(sql, rows)

    pos = (df["sentiment_char"] == "P").sum()
    neg = (df["sentiment_char"] == "N").sum()
    log.info("추론 완료 | 긍정=%d 부정=%d 중립=%d", pos, neg, len(df)-pos-neg)


# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="KcBERT 감성분석")
    parser.add_argument("--mode", choices=["train", "infer", "both"],
                        default="both")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        train()
    if args.mode in ("infer", "both"):
        infer()


if __name__ == "__main__":
    main()
