# file: app.py
import os
import re
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import sessionmaker
import redis

# ------------- CONFIG -------------
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_TTL = 60 * 60 * 2  # 2 hours

# ------------- DB init -------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(bind=engine)
metadata = MetaData()

# ------------- Redis -------------
r = redis.from_url(REDIS_URL, decode_responses=True)

# ------------- FastAPI -------------
app = FastAPI(title="Traffic Rule Inference Core", version="1.0.0")

# ------------- SYNONYMS -------------
SYNONYMS = {
    "vượt": ["vượt", "vượt xe"],
    "vượt đèn đỏ": ["vượt đèn đỏ", "băng đèn đỏ", "qua đèn đỏ"],
    "xe máy": ["xe máy", "môtô", "motor", "xe may"],
    "ô tô": ["ô tô", "oto", "xe hơi", "xe ô tô", "xe hoi"],
    "có biển cấm vượt": ["cấm vượt", "biển cấm vượt", "có biển cấm vượt"]
}

def build_inverse_synonym_map(syn_map: Dict[str, List[str]]) -> Dict[str, str]:
    inv = {}
    for canonical, lst in syn_map.items():
        for v in lst:
            inv[v.lower()] = canonical
    return inv

INV_SYNS = build_inverse_synonym_map(SYNONYMS)

def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^0-9a-zàáâãèéêìíòóôõùúăđĩũơưẹẻẽởộảạỳỷỹ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_facts(text: str) -> Dict[str, List[str]]:
    t = normalize_text(text)
    facts: Dict[str, List[str]] = {}
    
    # Dùng regex mạnh hơn
    if re.search(r"vượt\s+đèn\s+đỏ|băng\s+đèn\s+đỏ|qua\s+đèn\s+đỏ", t):
        facts.setdefault("action", []).append("vượt đèn đỏ")
    elif "vượt" in t:
        facts.setdefault("action", []).append("vượt")
        
    if re.search(r"xe\s+máy|môtô|motor", t):
        facts.setdefault("vehicle_type", []).append("xe máy")
    if re.search(r"ô\s*tô|xe\s+hơi|oto", t):
        facts.setdefault("vehicle_type", []).append("ô tô")
        
    if re.search(r"cấm\s+vượt|biển\s+cấm", t):
        facts.setdefault("context", []).append("có biển cấm vượt")
        
    return facts

# ------------- RULE LOADER -------------
# THAY TOÀN BỘ HÀM load_rules_from_db() BẰNG HÀM NÀY (ĐÃ TEST 100%)
def load_rules_from_db() -> List[Dict[str, Any]]:
    db = SessionLocal()
    rules = []
    try:
        print("Reflecting tables...")
        metadata.reflect(bind=engine, only=['rules', 'rule_conditions', 'penalties'])
        
        if 'rules' not in metadata.tables:
            print("Table 'rules' not found!")
            return []
            
        rules_table = metadata.tables['rules']
        cond_table = metadata.tables['rule_conditions']
        penalty_table = metadata.tables['penalties']

        print(f"Found tables: {list(metadata.tables.keys())}")

        # LẤY CHỈ active = 1
        q = select(rules_table).where(rules_table.c.active == 1)
        result = db.execute(q)
        rows = result.fetchall()

        print(f"Found {len(rows)} active rules in DB")

        for row in rows:
            # DÙNG row._mapping ĐỂ TRÁNH LỖI dict()
            rule_dict = dict(row._mapping)
            rule_id = rule_dict["id"]
            
            # Load conditions
            cond_q = select(cond_table).where(cond_table.c.rule_id == rule_id)
            cond_result = db.execute(cond_q)
            conds = {}
            for crow in cond_result.fetchall():
                attr = crow._mapping["attribute"]
                val = crow._mapping["value"]
                
                # XỬ LÝ VALUE: JSON hoặc CSV
                try:
                    if val and val.strip().startswith('['):
                        vals = json.loads(val)
                    else:
                        vals = [v.strip() for v in val.split(",") if v.strip()]
                except:
                    vals = [str(val).strip()] if val else []
                    
                if not isinstance(vals, list):
                    vals = [vals]
                conds.setdefault(attr, []).extend(vals)
            
            rule_dict["conditions_resolved"] = conds

            # Load penalty
            if rule_dict.get("penalty_id"):
                p_result = db.execute(select(penalty_table).where(penalty_table.c.id == rule_dict["penalty_id"])).fetchone()
                if p_result:
                    rule_dict["penalty_meta"] = dict(p_result._mapping)

            # Thêm priority nếu có
            rule_dict["priority"] = rule_dict.get("priority", 0)
            
            rules.append(rule_dict)

        print(f"SUCCESS: Loaded {len(rules)} rules into memory")
        return rules

    except Exception as e:
        print(f"CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        db.close()
# ------------- GLOBAL CACHE -------------
RULES_CACHE: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    global RULES_CACHE
    try:
        print("Connecting to MySQL...")
        print(f"DATABASE_URL: {DATABASE_URL}")  # In ra để kiểm tra
        RULES_CACHE = load_rules_from_db()
        print(f"Loaded {len(RULES_CACHE)} active rules")
    except Exception as e:
        print(f"FAILED TO LOAD RULES: {e}")
        RULES_CACHE = []

# ------------- INFERENCE ENGINE -------------
def forward_filter(facts: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    if not RULES_CACHE:
        return []
    candidates = []
    for rule in RULES_CACHE:
        conds = rule.get("conditions_resolved", {})
        for slot, required in conds.items():
            if slot in facts and any(req in facts[slot] for req in required):
                candidates.append(rule)
                break
    return candidates

def backward_check(rule: Dict[str, Any], facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    missing = {}
    conds = rule.get("conditions_resolved", {})
    for slot, required in conds.items():
        if slot not in facts or not any(req in facts[slot] for req in required):
            missing[slot] = required
    return missing

# ------------- QUESTIONS -------------
SLOT_QUESTION_TEMPLATES = {
    "vehicle_type": "Bạn đang điều khiển loại xe gì? (ô tô / xe máy)",
    "action": "Bạn đã làm gì vậy? (vượt đèn đỏ, vượt ẩu, đi ngược chiều...)",
    "context": "Có biển báo cấm vượt hay tình huống đặc biệt nào không?"
}

# ------------- MODELS -------------
class InferRequest(BaseModel):
    session_id: str
    text: Optional[str] = None
    facts: Optional[Dict[str, List[str]]] = None

class InferResponse(BaseModel):
    status: str
    questions: Optional[List[Dict[str, Any]]] = None
    results: Optional[List[Dict[str, Any]]] = None

# ------------- SESSION -------------
def get_session(sid: str) -> Dict[str, List[str]]:
    raw = r.get(f"session:{sid}")
    return json.loads(raw) if raw else {}

def save_session(sid: str, facts: Dict[str, List[str]]):
    r.setex(f"session:{sid}", REDIS_TTL, json.dumps(facts))

# ------------- ENDPOINT -------------
@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    print(f"Received infer request for session {req.session_id} with text: {req.text} and facts: {req.facts}")
    session_facts = get_session(req.session_id)
    new_facts = {}

    if req.text:
        new_facts = extract_facts(req.text)

    if req.facts:
        for k, v in req.facts.items():
            new_facts.setdefault(k, []).extend(v)

    # Merge
    merged = session_facts.copy()
    for k, v in new_facts.items():
        merged.setdefault(k, []).extend(v)
        merged[k] = list(set(merged[k]))  # dedup

    save_session(req.session_id, merged)

    candidates = forward_filter(merged)
    if not candidates:
        return InferResponse(
            status="unknown",
            questions=[{"question": "Mình chưa hiểu rõ. Bạn kể chi tiết hơn nhé! Ví dụ: 'tôi vượt đèn đỏ bằng xe máy'"}]
        )

    results = []
    missing_agg = {}

    for rule in candidates:
        miss = backward_check(rule, merged)
        if not miss:
            results.append({
                "rule_id": rule["id"],
                "code": rule.get("code"),
                "title": rule.get("title"),
                "penalty": rule.get("penalty_meta"),
                "conclusion": rule.get("conclusion")
            })
        else:
            for slot, opts in miss.items():
                missing_agg.setdefault(slot, set()).update(opts[:5])

    if results:
        return InferResponse(status="result", results=results)

    questions = [
        {
            "slot": slot,
            "question": SLOT_QUESTION_TEMPLATES.get(slot, f"Cho mình biết về {slot} nhé?"),
            "options": list(opts)
        }
        for slot, opts in missing_agg.items()
    ]
    return InferResponse(status="need_info", questions=questions)

@app.post("/reset/{session_id}")
def reset(session_id: str):
    r.delete(f"session:{session_id}")
    return {"ok": True}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rules_loaded": len(RULES_CACHE),
        "redis_connected": r.ping()
    }