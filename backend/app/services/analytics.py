from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from backend.app import models


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates the Word Error Rate (Levenshtein distance at the word level)."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0

    ref_words = [w.strip().lower() for w in reference.split() if w.strip()]
    hyp_words = [w.strip().lower() for w in hypothesis.split() if w.strip()]

    r_len = len(ref_words)
    h_len = len(hyp_words)

    # Initialize DP matrix
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    return dp[r_len][h_len] / r_len


def calculate_ner_metrics(predicted: list[str], ground_truth: list[str]) -> tuple[float, float, float]:
    """Calculates Precision, Recall, and F1 score for entity extraction."""
    pred_set = set([p.strip().lower() for p in predicted if p.strip()])
    gt_set = set([g.strip().lower() for g in ground_truth if g.strip()])

    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not gt_set:
        return 0.0, 1.0, 0.0
    if not pred_set:
        return 1.0, 0.0, 0.0

    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def log_consultation_analytics(
    db: Session,
    consultation_id: int,
    doctor_id: int,
    audio_duration: float,
    processing_time: float,
    transcription_time: float,
    summary_time: float,
) -> models.ConsultationAnalytics:
    """Logs detailed consultation and timing metrics to the database."""
    entry = models.ConsultationAnalytics(
        consultation_id=consultation_id,
        doctor_id=doctor_id,
        audio_duration=audio_duration,
        processing_time=processing_time,
        transcription_time=transcription_time,
        summary_time=summary_time,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def log_ai_evaluation(
    db: Session,
    consultation_id: int,
    wer: float,
    precision: float,
    recall: float,
    f1_score: float,
    summary_score: float = None,
) -> models.AIEvaluation:
    """Logs calculated evaluation and quality scores to the database."""
    entry = models.AIEvaluation(
        consultation_id=consultation_id,
        wer=wer,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        summary_score=summary_score,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def update_doctor_analytics(db: Session, doctor_id: int, specialization: str = "General Practice"):
    """Updates Doctor login frequency, active days, and aggregate consultation count."""
    analytics = db.query(models.DoctorAnalytics).filter(models.DoctorAnalytics.doctor_id == doctor_id).first()
    if not analytics:
        analytics = models.DoctorAnalytics(
            doctor_id=doctor_id,
            specialization=specialization,
            consultations_count=1,
            active_days=1,
        )
        db.add(analytics)
    else:
        analytics.consultations_count += 1
        # Simple active days heuristic: if last login is older than 20 hours, increment active days
        if analytics.last_login:
            time_diff = datetime.utcnow() - analytics.last_login.replace(tzinfo=None)
            if time_diff > timedelta(hours=20):
                analytics.active_days += 1
        analytics.last_login = func.now()
    db.commit()
