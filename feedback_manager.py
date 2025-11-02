
import os, json
from datetime import datetime, timezone

FEEDBACK_FILE = os.path.join("data", "feedback.json")

def ensure_feedback_file():
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def log_feedback(user, question, answer, helpful, comment=""):
    ensure_feedback_file()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "question": question,
        "helpful": bool(helpful),
        "comment": comment
    }
    with open(FEEDBACK_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

def read_feedback_stats():
    ensure_feedback_file()
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    helpful = sum(1 for d in data if d.get("helpful"))
    return {"total": total, "helpful": helpful, "not_helpful": total - helpful}
