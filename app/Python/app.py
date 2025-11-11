# file: app.py (PHIÊN BẢN HOÀN CHỈNH - PRODUCTION READY)
import os
import re
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import redis

# ============== CONFIG ==============
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_TTL = 60 * 60 * 24  # 24h

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, echo=False)
SessionLocal = sessionmaker(bind=engine)
r = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="Vietnam Traffic Law Inference Engine", version="2.0.0")

# ============== VIETNAMESE LEGAL NLP ==============
VIETNAMESE_KEYWORDS = {
    "action": {
        "vượt đèn đỏ": ["vượt đèn đỏ", "băng đèn đỏ", "qua đèn đỏ", "đèn đỏ mà đi", "vượt đèn", "đèn đỏ vẫn chạy"],
        "không đội mũ bảo hiểm": ["không đội mũ", "không mũ bảo hiểm", "không mbh", "chạy không mũ"],
        "đi ngược chiều": ["ngược chiều", "đi ngược đường", "chạy ngược", "đi ngược làn"],
        "vượt ẩu": ["vượt ẩu", "vượt xe nguy hiểm", "vượt bên phải", "cắt đầu xe"],
        "đi vào đường cấm": ["đi vào đường cấm", "vào đường cấm", "đi ngược chiều cấm"],
        "lạng lách": ["lạng lách", "đánh võng", "lách đánh võng"],
        "chở quá số người": ["chở 3", "chở 4", "chở quá người", "chở 3 người"],
    },
    "vehicle_type": {
        "xe máy": ["xe máy", "môtô", "motor", "xe may", "xe gắn máy"],
        "ô tô": ["ô tô", "oto", "xe hơi", "xe con", "xe tải", "xe khách"],
    },
    "context": {
        "có biển cấm vượt": ["có biển cấm vượt", "biển cấm vượt", "cấm vượt", "biển 104"],
        "đường cao tốc": ["cao tốc", "đường cao tốc"],
        "khu vực đông dân cư": ["khu dân cư", "đông dân", "trong phố"],
        "ban đêm": ["ban đêm", "tối", "đêm khuya"],
    }
}

# Build reverse map
CANONICAL_MAP = {}
for slot, groups in VIETNAMESE_KEYWORDS.items():
    for canonical, phrases in groups.items():
        for p in phrases:
            CANONICAL_MAP[p.lower()] = (slot, canonical)

def extract_facts_smart(text: str) -> Dict[str, List[str]]:
    text = text.lower()
    facts = {}
    
    for phrase, (slot, canonical) in CANONICAL_MAP.items():
        if phrase in text:
            facts.setdefault(slot, []).append(canonical)
    
    # Đặc biệt: nếu có "xe máy" mà không có vehicle_type → thêm
    # if any(x in text for x in ["xe máy", "môtô", "motor"]):
    #     facts.setdefault("vehicle_type", []).append("xe máy")
    # if any(x in text for x in ["ô tô", "oto", "xe hơi"]):
    #     facts.setdefault("vehicle_type", []).append("ô tô")
        
    # Dedup
    for k in facts:
        facts[k] = list(set(facts[k]))
    return facts

# ============== LOAD RULES (SIÊU ỔN ĐỊNH) ==============
def load_rules_from_db() -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        from sqlalchemy import MetaData
        metadata = MetaData()
        metadata.reflect(bind=engine, only=['rules', 'rule_conditions', 'penalties'])
        
        rules_table = metadata.tables['rules']
        cond_table = metadata.tables['rule_conditions']
        penalty_table = metadata.tables['penalties']

        query = select(rules_table).where(rules_table.c.active == 1)
        rows = db.execute(query).fetchall()

        rules = []
        for row in rows:
            rule = dict(row._mapping)
            rule_id = rule["id"]

            # Load conditions
            cond_q = select(cond_table).where(cond_table.c.rule_id == rule_id)
            conds = {}
            for crow in db.execute(cond_q).fetchall():
                attr = crow.attribute
                val = crow.value
                try:
                    values = json.loads(val) if val.strip().startswith('[') else [v.strip() for v in val.split(',') if v.strip()]
                except:
                    values = [val.strip()] if val else []
                conds.setdefault(attr, []).extend(values)
            
            rule["conditions"] = conds
            rule["priority"] = int(rule.get("priority", 0))

            # Load penalty
            if rule.get("penalty_id"):
                p = db.execute(select(penalty_table).where(penalty_table.c.id == rule["penalty_id"])).fetchone()
                print(f"Loading penalty for rule {rule_id}: {p}")
                if p:
                    penalty = dict(p._mapping)
                    fine_min = int(float(penalty.get("fine_min", 0))) if penalty.get("fine_min") else 0
                    fine_max = int(float(penalty.get("fine_max", 0))) if penalty.get("fine_max") else 0
                    rule["penalty"] = {
                        "amount": f"{fine_min:,}-{fine_max:,}" if fine_min and fine_max else f"{fine_max or fine_min:,}",
                        "unit": penalty.get("unit", "đồng"),
                        "additional": penalty.get("additional_punishment", ""),
                        "legal_ref": penalty.get("law_ref", "")
                    }

            rules.append(rule)
        
        # Sắp xếp theo priority (cao hơn = nặng hơn)
        rules.sort(key=lambda x: x["priority"], reverse=True)
        print(f"Loaded {len(rules)} rules (sorted by priority)")
        return rules

    except Exception as e:
        print(f"ERROR loading rules: {e}")
        traceback.print_exc()
        return []
    finally:
        db.close()

RULES = []
@app.on_event("startup")
async def startup():
    global RULES
    RULES = load_rules_from_db()
    if not RULES:
        print("CRITICAL: NO RULES LOADED!")

# ============== INFERENCE ENGINE ==============
def forward_chaining(facts: Dict[str, List[str]]) -> List[Dict]:
    matched = []
    for rule in RULES:
        conds = rule["conditions"]
        match = True
        for slot, required in conds.items():
            if slot not in facts or not any(req in facts[slot] for req in required):
                match = False
                break
        if match:
            matched.append(rule)
    return matched

# THAY TOÀN BỘ hàm find_missing_conditions() bằng cái này
# THAY TOÀN BỘ HÀM NÀY (từ dòng ~170)
def find_missing_conditions(facts: Dict[str, List[str]], session_id: str) -> Dict[str, List[str]]:
    missing = {}
    candidate_rules = []
    
    # Tìm các rule tiềm năng (đã khớp ít nhất 1 điều kiện)
    for rule in RULES:
        conds = rule["conditions"]
        satisfied = sum(1 for slot, reqs in conds.items() 
                       if slot in facts and any(r in facts[slot] for r in reqs))
        total = len(conds)
        if satisfied > 0:
            candidate_rules.append((rule, satisfied / total))
    
    # Sắp xếp theo độ khớp cao nhất
    candidate_rules.sort(key=lambda x: x[1], reverse=True)
    
    # LẤY DANH SÁCH SLOT ĐÃ HỎI TRƯỚC ĐÓ
    asked_key = f"asked:{session_id}"
    asked_slots = set(r.smembers(asked_key))
    
    # Chỉ hỏi những slot CHƯA hỏi
    for rule, score in candidate_rules[:3]:  # chỉ xét 3 rule tốt nhất
        for slot, reqs in rule["conditions"].items():
            if slot in asked_slots:
                continue
            if slot not in facts or not any(r in facts[slot] for r in reqs):
                missing.setdefault(slot, set()).update(reqs[:5])  # giới hạn 5 gợi ý
    
    # LƯU LẠI NHỮNG SLOT SẮP HỎI ĐỂ LẦN SAU KHÔNG LẶP
    if missing:
        r.sadd(asked_key, *missing.keys())
        r.expire(asked_key, REDIS_TTL)
    
    return {k: list(v) for k, v in missing.items()}
# ============== SMART QUESTIONS ==============
SMART_QUESTIONS = {
    "vehicle_type": {
        "question": "Bạn đang điều khiển loại xe nào?",
        "options": ["xe máy", "ô tô", "xe tải", "xe khách", "xe đạp"]
    },
    "action": {
        "question": "Bạn đã làm hành vi gì?",
        "options": ["vượt đèn đỏ", "không đội mũ bảo hiểm", "đi ngược chiều", "vượt ẩu", "lạng lách", "chở quá người"]
    },
    "context": {
        "question": "Có tình huống đặc biệt nào không?",
        "options": ["có biển cấm vượt", "đường cao tốc", "ban đêm", "khu dân cư đông", "không có gì đặc biệt"]
    }
}

# ============== API MODELS ==============
class QueryRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    text: Optional[str] = None  # THÊM DÒNG NÀY
    facts: Optional[Dict[str, List[str]]] = None

    def get_input_text(self) -> Optional[str]:
        return self.message or self.text
class QueryResponse(BaseModel):
    status: str  # "result" | "need_info" | "unknown"
    message: str = None
    violations: List[Dict] = None
    questions: List[Dict] = None
    session_facts: Dict[str, List[str]] = None

# ============== SESSION ==============
def get_facts(sid: str) -> Dict[str, List[str]]:
    data = r.get(f"session:{sid}")
    return json.loads(data) if data else {}

def save_facts(sid: str, facts: Dict[str, List[str]]):
    r.setex(f"session:{sid}", REDIS_TTL, json.dumps(facts))

# ============== MAIN ENDPOINT ==============
@app.post("/infer", response_model=QueryResponse)
async def infer(req: QueryRequest):
    if req.message and any(req.message.lower().startswith(x) for x in ["tôi", "em", "mình", "tớ"]):
        current_facts = {}  # coi như mô tả mới hoàn toàn
        r.delete(f"asked:{req.session_id}")
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    print(f"[INFER] session={req.session_id} message={req.message} text={req.text} facts={req.facts}")
    current_facts = get_facts(req.session_id)
    new_facts = {}

    input_text = req.get_input_text()
    if input_text:
        new_facts = extract_facts_smart(input_text)
    print(f"Extracted facts from text: {new_facts}")
    if req.facts:
        for k, v in req.facts.items():
            new_facts.setdefault(k, []).extend(v)

    # Merge + dedup
    for k, v in new_facts.items():
        current_facts.setdefault(k, []).extend(v)
        current_facts[k] = list(set(current_facts[k]))

    print(f"Current facts: {current_facts}")
    save_facts(req.session_id, current_facts)
    if req.message:
        r.delete(f"asked:{req.session_id}")
    # Kiểm tra có vi phạm không
    violations = forward_chaining(current_facts)
    for v in violations:
        print(f"Matched rule: {v['id']} with priority {v.get('priority')}")

    if violations:
        results = []
        for v in violations:
            results.append({
                "id": v["code"] or v["id"],
                "title": v.get("title", "Vi phạm giao thông"),
                "legal_ref": v["penalty"].get("legal_ref", "Nghị định 168/2024"),
                "penalty": f"{v['penalty']['amount']} {v['penalty']['unit']}",
                "additional": v['penalty'].get("additional", ""),
                "description": v.get("conclusion", "Bạn đã vi phạm luật giao thông đường bộ")
            })
        
        return QueryResponse(
            status="result",
            message=f"Đã phát hiện {len(violations)} hành vi vi phạm!",
            violations=results,
            session_facts=current_facts
        )

    # Nếu chưa đủ dữ liệu → hỏi lại
    missing = find_missing_conditions(current_facts, req.session_id)
    if missing:
        questions = []
        for slot in missing.keys():
            template = SMART_QUESTIONS.get(slot, {
                "question": f"Cho mình biết về {slot} nhé?",
                "options": missing[slot][:5]
            })
            questions.append({
                "slot": slot,
                "question": template["question"],
                "options": template["options"]
            })
        print(questions, current_facts)
        return QueryResponse(
            status="need_info",
            message="Mình cần thêm thông tin để tư vấn chính xác!",
            questions=questions,
            session_facts=current_facts
        )

    # Không hiểu gì cả
    return QueryResponse(
        status="unknown",
        message="Mình chưa hiểu bạn vi phạm lỗi gì. Hãy kể rõ hơn nhé!\nVí dụ: 'Tôi vượt đèn đỏ bằng xe máy, không đội mũ bảo hiểm'",
        session_facts=current_facts
    )

# ============== UTILS ==============
@app.post("/reset/{session_id}")
def reset(session_id: str):
    r.delete(f"session:{session_id}")
    return {"status": "reset"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rules_count": len(RULES),
        "redis": r.ping(),
        "db": "connected" if RULES else "no rules"
    }