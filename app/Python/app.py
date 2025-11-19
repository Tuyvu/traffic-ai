# file: app.py (PHIÊN BẢN HOÀN CHỈNH VỚI EMBEDDING)
import os
import re
import json
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import redis
from sentence_transformers import SentenceTransformer
import torch

# ============== CONFIG ==============
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_DATABASE')}"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_TTL = 60 * 60 * 24  # 24h

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, echo=False)
SessionLocal = sessionmaker(bind=engine)
r = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="Vietnam Traffic Law Inference Engine", version="3.0.0")

# ============== EMBEDDING MODEL ==============
MODEL_NAME = "keepitreal/vietnamese-sbert"
model = None

@app.on_event("startup")
async def startup():
    global model, RULES, RULE_CONDITIONS
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully!")
    
    RULES, RULE_CONDITIONS = load_rules_from_db()
    if not RULES:
        print("CRITICAL: NO RULES LOADED!")

# ============== EMBEDDING UTILS ==============
def get_embedding(text: str) -> List[float]:
    """Generate embedding for text"""
    if not text or not text.strip():
        return [0.0] * 384  # Default dimension
    
    with torch.no_grad():
        embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar_rules(query_embedding: List[float], threshold: float = 0.7) -> List[Dict]:
    """Find rules with similar conditions using embedding similarity"""
    similar_rules = []
    
    for rule_cond in RULE_CONDITIONS:
        similarity = cosine_similarity(query_embedding, rule_cond["embedding"])
        if similarity >= threshold:
            # Find the rule details
            rule = next((r for r in RULES if r["id"] == rule_cond["rule_id"]), None)
            if rule and rule not in similar_rules:
                similar_rules.append({
                    "rule": rule,
                    "similarity": similarity,
                    "matched_condition": rule_cond
                })
    
    # Sort by similarity score
    similar_rules.sort(key=lambda x: x["similarity"], reverse=True)
    return similar_rules

# ============== LOAD RULES WITH EMBEDDING ==============
ATTRIBUTES_MAPPING = {}
RULE_CONDITIONS = []

def load_rules_from_db() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    global ATTRIBUTES_MAPPING, RULE_CONDITIONS
    db = SessionLocal()
    try:
        from sqlalchemy import MetaData
        metadata = MetaData()
        metadata.reflect(bind=engine, only=['rules', 'rule_conditions', 'penalties'])
        
        rules_table = metadata.tables['rules']
        cond_table = metadata.tables['rule_conditions']
        penalty_table = metadata.tables['penalties']

        # Load all active rules
        query = select(rules_table).where(rules_table.c.active == 1)
        rows = db.execute(query).fetchall()

        rules = []
        rule_conditions = []
        
        for row in rows:
            rule = dict(row._mapping)
            rule_id = rule["id"]

            # Load conditions with embeddings
            cond_q = select(cond_table).where(cond_table.c.rule_id == rule_id)
            conds = {}
            for crow in db.execute(cond_q).fetchall():
                attr = crow.attribute
                val = crow.value
                embedding = crow.embedding
                
                print(f"[LOAD_COND] Rule {rule_id}: attr={attr}, val={val!r}")
                try:
                    if embedding:
                        # Parse embedding from database (assuming it's stored as JSON string)
                        embedding_vec = json.loads(embedding)
                    else:
                        # Fallback: generate embedding from attribute + value
                        text_to_embed = f"{attr} {val}"
                        embedding_vec = get_embedding(text_to_embed)
                except Exception as e:
                    print(f"[LOAD_COND] ERROR processing embedding: {e}")
                    text_to_embed = f"{attr} {val}"
                    embedding_vec = get_embedding(text_to_embed)
                
                # Store condition with embedding
                rule_cond = {
                    "rule_id": rule_id,
                    "attribute": attr,
                    "value": val,
                    "embedding": embedding_vec
                }
                rule_conditions.append(rule_cond)
                
                # Parse values for traditional matching
                try:
                    values = json.loads(val) if val.strip().startswith('[') else [v.strip() for v in val.split(',') if v.strip()]
                except Exception as e:
                    print(f"[LOAD_COND] ERROR parsing values: {e}")
                    values = [val.strip()] if val else []
                
                values = [v.lower() for v in values]
                conds.setdefault(attr, []).extend(values)
                
                # Build attributes mapping
                for v in values:
                    ATTRIBUTES_MAPPING[v] = attr
            
            rule["conditions"] = conds
            rule["priority"] = int(rule.get("priority", 0))

            # Load penalty
            if rule.get("penalty_id"):
                p = db.execute(select(penalty_table).where(penalty_table.c.id == rule["penalty_id"])).fetchone()
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
        
        # Sort rules by priority
        rules.sort(key=lambda x: x["priority"], reverse=True)
        
        print(f"\nLoaded {len(rules)} rules and {len(rule_conditions)} conditions with embeddings")
        return rules, rule_conditions

    except Exception as e:
        print(f"ERROR loading rules: {e}")
        traceback.print_exc()
        return [], []
    finally:
        db.close()

RULES = []

# ============== VIETNAMESE LEGAL NLP ==============
VIETNAMESE_KEYWORDS = {
    "action": {
        "vượt đèn đỏ": ["vượt đèn đỏ", "băng đèn đỏ", "qua đèn đỏ", "đèn đỏ mà đi", "vượt đèn", "đèn đỏ vẫn chạy"],
        "không đội mũ bảo hiểm": ["không đội mũ", "không mũ bảo hiểm", "không mbh", "chạy không mũ"],
        "đi ngược chiều": ["ngược chiều", "đi ngược đường", "chạy ngược", "đi ngược làn"],
        "vượt ẩu": ["vượt ẩu", "vượt xe nguy hiểm", "vượt bên phải", "cắt đầu xe"],
        "lạng lách": ["lạng lách", "đánh võng", "lách đánh võng"],
        "chở quá người": ["chở 3", "chở 4", "chở quá người", "chở 3 người"],
        "dừng đỗ sai quy định": ["dừng sai chỗ", "đỗ sai nơi quy định", "dừng đỗ trái phép"],
        "chuyển hướng không an toàn": ["chuyển hướng không quan sát", "rẽ không báo hiệu", "quay đầu không an toàn"],
        "chuyển làn không đúng": ["chuyển làn không báo hiệu", "sang làn ẩu", "chuyển nhiều làn"],
        "chạy quá tốc độ": ["chạy quá tốc độ", "vượt quá tốc độ", "phóng nhanh"],
        "chạy tốc độ thấp gây cản trở": ["chạy quá chậm", "tốc độ thấp gây cản trở"],
        "không xử lý tai nạn": ["gây tai nạn bỏ chạy", "không dừng lại sau tai nạn"],
        "đi vào đường cấm": ["đi vào đường cấm", "vào đường cấm", "đi ngược chiều cấm"],
        "điều khiển xe 1 bánh/2 bánh": ["chạy một bánh", "chạy hai bánh", "wheelie"],
        "điều khiển xe dàn hàng ngang": ["dàn hàng ngang", "chạy hàng ngang"],
        "kéo theo xe khác": ["kéo theo xe", "kéo xe khác"],
        "gây tai nạn": ["gây tai nạn", "đâm va tai nạn"],
        "điều khiển xe thành đoàn": ["đi thành đoàn", "chạy đoàn xe"],
        "nhóm xe chạy quá tốc độ": ["nhóm xe chạy nhanh", "cả nhóm vượt tốc độ"],
        "bỏ chạy sau tai nạn": ["bỏ chạy sau tai nạn", "tai nạn rồi bỏ đi"],
        "không chấp hành biển báo": ["không tuân theo biển báo", "vượt biển cấm"],
        "không có tín hiệu khi vượt": ["vượt không báo hiệu", "vượt không xi nhan"]
    },
    "vehicle_type": {
        "xe máy": ["xe máy", "môtô", "motor", "xe may", "xe gắn máy"],
        "ô tô": ["ô tô", "oto", "xe hơi", "xe con", "xe tải", "xe khách"],
        "xe đạp": ["xe đạp", "xe đạp điện"],
        "xe ba bánh": ["xe ba bánh", "xích lô"]
    },
    "context": {
        "có biển cấm vượt": ["có biển cấm vượt", "biển cấm vượt", "cấm vượt", "biển 104"],
        "đường cao tốc": ["cao tốc", "đường cao tốc"],
        "ban đêm": ["ban đêm", "tối", "đêm khuya"],
        "khu dân cư đông": ["khu dân cư", "đông dân", "trong phố"],
        "đường một chiều": ["đường một chiều", "một chiều"],
        "đường cấm ngược chiều": ["đường cấm ngược chiều", "cấm ngược chiều"],
        "khu vực cấm": ["khu vực cấm", "vào khu vực cấm"],
        "đường cong không giao nhau": ["đường cong", "đường vòng"],
        "trên cầu": ["trên cầu", "cầu"],
        "điểm đón trả khách": ["điểm đón khách", "bến xe bus"],
        "nơi đường giao nhau": ["ngã tư", "ngã ba", "giao lộ"],
        "phần đường người đi bộ": ["vạch người đi bộ", "làn người đi bộ"],
        "hầm đường bộ": ["trong hầm", "hầm đường bộ"],
        "lề đường": ["lề đường", "vạt đường"]
    },
    "so_nguoi_ngoi_sau": {
        "1 người": ["1 người", "một mình"],
        "2 người": ["2 người", "hai người"],
        "3 người": ["3 người", "ba người"],
        "4 người trở lên": ["4 người", "nhiều người", "trên 3 người"]
    },
    "vi_tri_dung_do": {
        "lòng đường": ["lòng đường", "giữa đường"],
        "vỉa hè": ["vỉa hè", "hè phố"],
        "trên cầu": ["trên cầu", "cầu"],
        "điểm đón trả khách": ["điểm đón khách", "bến xe bus"],
        "nơi đường giao nhau": ["ngã tư", "ngã ba", "giao lộ"],
        "phần đường người đi bộ": ["vạch người đi bộ", "làn người đi bộ"],
        "hầm đường bộ": ["trong hầm", "hầm đường bộ"],
        "đường cao tốc": ["cao tốc", "đường cao tốc"],
        "nơi có biển cấm dừng đỗ": ["biển cấm dừng", "biển cấm đỗ"]
    },
    "hanh_dong_chuyen_huong": {
        "không quan sát": ["không quan sát", "không nhìn"],
        "không báo hiệu": ["không báo hiệu", "không xi nhan"],
        "không giảm tốc độ": ["không giảm tốc", "vẫn giữ tốc độ"],
        "chuyển hướng đột ngột": ["chuyển hướng đột ngột", "rẽ đột ngột"],
        "chuyển hướng an toàn": ["chuyển hướng an toàn", "rẽ an toàn"]
    },
    "hanh_dong_chuyen_lan": {
        "không báo hiệu": ["không báo hiệu", "không xi nhan"],
        "chuyển nhiều làn cùng lúc": ["chuyển nhiều làn", "sang nhiều làn"],
        "chuyển làn sai nơi quy định": ["sai nơi quy định", "chuyển làn ẩu"],
        "chuyển làn an toàn": ["chuyển làn an toàn", "sang làn đúng"]
    },
    "huong_di_chuyen": {
        "đúng chiều": ["đúng chiều", "chiều đúng"],
        "ngược chiều": ["ngược chiều", "sai chiều"],
        "trên vỉa hè": ["trên vỉa hè", "lên vỉa hè"],
        "trên lề đường": ["trên lề đường", "lề đường"]
    },
    "loai_duong": {
        "đường trong khu dân cư": ["khu dân cư", "trong phố"],
        "đường cao tốc": ["cao tốc", "đường cao tốc"],
        "đường một chiều": ["đường một chiều", "một chiều"],
        "đường cấm ngược chiều": ["đường cấm ngược chiều", "cấm ngược chiều"],
        "đường đôi": ["đường đôi", "chia hai chiều"],
        "đường ngoại thành": ["ngoại thành", "đường liên tỉnh"]
    },
    "toc_do": {
        "dưới 40 km/h": ["chậm", "dưới 40"],
        "40-60 km/h": ["40-60", "trung bình"],
        "60-80 km/h": ["60-80", "khá nhanh"],
        "80-100 km/h": ["80-100", "nhanh"],
        "trên 100 km/h": ["trên 100", "rất nhanh"]
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
    matched_phrases = set()
    
    # Strategy 1: Exact substring match từ ATTRIBUTES_MAPPING
    for value, attr in ATTRIBUTES_MAPPING.items():
        if value in text:
            facts.setdefault(attr, []).append(value)
            matched_phrases.add(value)
    
    # Strategy 2: Fuzzy matching với similarity > 0.4
    best_matches = {}
    for value, attr in ATTRIBUTES_MAPPING.items():
        if value in matched_phrases:
            continue
        ratio = SequenceMatcher(None, value, text).ratio()
        if ratio > 0.4:
            if attr not in best_matches or best_matches[attr][1] < ratio:
                best_matches[attr] = (value, ratio)
    
    for attr, (value, ratio) in best_matches.items():
        facts.setdefault(attr, []).append(value)
    
    # Strategy 3: Keyword matching từ CANONICAL_MAP
    if not facts:
        for phrase, (slot, canonical) in CANONICAL_MAP.items():
            if phrase in text:
                facts.setdefault(slot, []).append(canonical)
    
    # Dedup
    for k in facts:
        facts[k] = list(set(facts[k]))
    
    print(f"[EXTRACT] text='{text}' -> facts={facts}")
    return facts

# ============== INFERENCE ENGINE ==============
def forward_chaining(facts: Dict[str, List[str]], candidate_rules: List[Dict] = None) -> List[Dict]:
    """
    Forward chaining: match rules dựa trên facts hiện có.
    Nếu có candidate_rules (từ embedding), chỉ xét các rule đó.
    """
    rules_to_check = candidate_rules if candidate_rules else RULES
    matched = []
    
    for rule in rules_to_check:
        conds = rule["conditions"]
        matched_conds = 0
        
        print(f"[FC] Checking Rule {rule['id']}: conditions={conds}")
        print(f"[FC]   Current facts={facts}")
        
        for slot, required in conds.items():
            print(f"[FC]   Checking slot '{slot}': required={required}")
            if slot in facts:
                print(f"[FC]     Slot exists in facts: {facts[slot]}")
                for req in required:
                    if req in facts[slot]:
                        print(f"[FC]       MATCH: '{req}'")
                        matched_conds += 1
                        break
                    else:
                        print(f"[FC]       NO MATCH: '{req}' NOT in {facts[slot]}")
            else:
                print(f"[FC]     Slot '{slot}' NOT in facts")
        
        if matched_conds > 0:
            print(f"[FC] ✓ Rule {rule['id']}: matched {matched_conds}/{len(conds)} conditions")
            matched.append(rule)
        else:
            print(f"[FC] ✗ Rule {rule['id']}: no conditions matched")
    
    return matched

def find_missing_conditions(facts: Dict[str, List[str]], session_id: str, candidate_rules: List[Dict] = None) -> Dict[str, List[str]]:
    """Tìm điều kiện còn thiếu từ các rule khả thi"""
    rules_to_check = candidate_rules if candidate_rules else RULES
    missing = {}
    candidate_rules_scored = []
    
    # Tìm các rule tiềm năng (đã khớp ít nhất 1 điều kiện)
    for rule in rules_to_check:
        conds = rule["conditions"]
        satisfied = sum(1 for slot, reqs in conds.items() 
                       if slot in facts and any(r in facts[slot] for r in reqs))
        total = len(conds)
        if satisfied > 0:
            candidate_rules_scored.append((rule, satisfied / total))
    
    # Sắp xếp theo độ khớp cao nhất
    candidate_rules_scored.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy danh sách slot đã hỏi trước đó
    asked_key = f"asked:{session_id}"
    asked_slots = set(r.smembers(asked_key))
    
    # Chỉ hỏi những slot CHƯA hỏi từ các rule tốt nhất
    for rule, score in candidate_rules_scored[:3]:
        for slot, reqs in rule["conditions"].items():
            # Skip nếu đã có dữ liệu cho slot này
            if slot in facts and any(r in facts[slot] for r in reqs):
                continue
            # Skip nếu đã hỏi lần trước
            if slot in asked_slots:
                continue
            # Thêm vào danh sách cần hỏi
            missing.setdefault(slot, set()).update(reqs[:5])
    
    # Lưu lại những slot sắp hỏi
    if missing:
        r.sadd(asked_key, *missing.keys())
        r.expire(asked_key, REDIS_TTL)
    
    return {k: list(v) for k, v in missing.items()}

# ============== SMART QUESTIONS ==============
SMART_QUESTIONS = {
    "action": {
        "question": "Bạn đã làm hành vi gì?",
        "options": [
            "vượt đèn đỏ", "không đội mũ bảo hiểm", "đi ngược chiều", "vượt ẩu", 
            "lạng lách", "chở quá người", "dừng đỗ sai quy định", "chuyển hướng không an toàn",
            "chuyển làn không đúng", "chạy quá tốc độ", "chạy tốc độ thấp gây cản trở",
            "không xử lý tai nạn", "đi vào đường cấm", "điều khiển xe 1 bánh/2 bánh",
            "điều khiển xe dàn hàng ngang", "kéo theo xe khác", "gây tai nạn",
            "điều khiển xe thành đoàn", "nhóm xe chạy quá tốc độ", "bỏ chạy sau tai nạn",
            "không chấp hành biển báo", "không có tín hiệu khi vượt"
        ]
    },
    "vehicle_type": {
        "question": "Bạn đang điều khiển loại xe nào?",
        "options": ["xe máy", "ô tô", "xe tải", "xe khách", "xe đạp", "xe ba bánh"]
    },
    "context": {
        "question": "Có tình huống đặc biệt nào không?",
        "options": [
            "có biển cấm vượt", "đường cao tốc", "ban đêm", "khu dân cư đông", 
            "không có gì đặc biệt", "đường một chiều", "đường cấm ngược chiều",
            "khu vực cấm", "đường cong không giao nhau", "trên cầu",
            "điểm đón trả khách", "nơi đường giao nhau", "phần đường người đi bộ",
            "hầm đường bộ", "lề đường"
        ]
    },
    "so_nguoi_ngoi_sau": {
        "question": "Xe của bạn đang chở bao nhiêu người (tính cả người lái)?",
        "options": ["1 người", "2 người", "3 người", "4 người trở lên"]
    },
    "vi_tri_dung_do": {
        "question": "Bạn dừng/đỗ xe ở đâu?",
        "options": [
            "lòng đường", "vỉa hè", "trên cầu", "điểm đón trả khách", 
            "nơi đường giao nhau", "phần đường người đi bộ", "hầm đường bộ",
            "đường cao tốc", "nơi có biển cấm dừng đỗ"
        ]
    },
    "hanh_dong_chuyen_huong": {
        "question": "Bạn đã chuyển hướng như thế nào?",
        "options": [
            "không quan sát", "không báo hiệu", "không giảm tốc độ", 
            "chuyển hướng đột ngột", "chuyển hướng an toàn"
        ]
    },
    "hanh_dong_chuyen_lan": {
        "question": "Bạn đã chuyển làn như thế nào?",
        "options": [
            "không báo hiệu", "chuyển nhiều làn cùng lúc", "chuyển làn sai nơi quy định",
            "chuyển làn an toàn"
        ]
    },
    "huong_di_chuyen": {
        "question": "Bạn đang đi như thế nào?",
        "options": [
            "đúng chiều", "ngược chiều", "trên vỉa hè", "trên lề đường"
        ]
    },
    "loai_duong": {
        "question": "Bạn đang đi trên loại đường nào?",
        "options": [
            "đường trong khu dân cư", "đường cao tốc", "đường một chiều", 
            "đường cấm ngược chiều", "đường đôi", "đường ngoại thành"
        ]
    },
    "toc_do": {
        "question": "Bạn đang chạy với tốc độ bao nhiêu?",
        "options": [
            "dưới 40 km/h", "40-60 km/h", "60-80 km/h", "80-100 km/h", "trên 100 km/h"
        ]
    }
}

# ============== API MODELS ==============
class QueryRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    text: Optional[str] = None
    facts: Optional[Dict[str, List[str]]] = None

    def get_input_text(self) -> Optional[str]:
        return self.message or self.text

class QueryResponse(BaseModel):
    status: str  # "result" | "need_info" | "unknown"
    message: str = None
    violations: List[Dict] = Field(default_factory=list)
    questions: List[Dict] = Field(default_factory=list)
    session_facts: Dict[str, List[str]] = Field(default_factory=dict)
    candidate_rules_count: int = 0  # Thêm thông tin debug

# ============== SESSION ==============
def get_facts(sid: str) -> Dict[str, List[str]]:
    data = r.get(f"session:{sid}")
    return json.loads(data) if data else {}

def save_facts(sid: str, facts: Dict[str, List[str]]):
    r.setex(f"session:{sid}", REDIS_TTL, json.dumps(facts))

# ============== MAIN ENDPOINT VỚI EMBEDDING ==============
@app.post("/infer", response_model=QueryResponse)
async def infer(req: QueryRequest):
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    
    print(f"[INFER] session={req.session_id} message={req.message} text={req.text}")
    
    current_facts = get_facts(req.session_id)
    new_facts = {}
    candidate_rules = []

    # BƯỚC 1: EMBEDDING CÂU HỎI VÀ SO KHỚP
    input_text = req.get_input_text()
    if input_text:
        # Extract facts bằng keyword matching
        new_facts = extract_facts_smart(input_text)
        
        # Generate embedding cho câu hỏi
        query_embedding = get_embedding(input_text)
        print(f"[EMBEDDING] Generated embedding for query: {len(query_embedding)} dimensions")
        
        # Tìm các rule khả thi bằng embedding
        similar_rules_info = find_similar_rules(query_embedding, threshold=0.7)
        candidate_rules = [info["rule"] for info in similar_rules_info]
        
        print(f"[EMBEDDING] Found {len(candidate_rules)} candidate rules via embedding")
        for info in similar_rules_info[:3]:  # Log top 3
            print(f"[EMBEDDING] Rule {info['rule']['id']} - similarity: {info['similarity']:.3f}")

    # Merge facts
    if req.facts:
        for k, v in req.facts.items():
            new_facts.setdefault(k, []).extend(v)

    for k, v in new_facts.items():
        current_facts.setdefault(k, []).extend(v)
        current_facts[k] = list(set(current_facts[k]))

    save_facts(req.session_id, current_facts)
    
    # Reset asked slots nếu có message mới
    if req.message:
        r.delete(f"asked:{req.session_id}")

    # BƯỚC 2: SUY DIỄN TIẾN VỚI CANDIDATE RULES
    if candidate_rules:
        violations = forward_chaining(current_facts, candidate_rules)
        print(f"[FC] Found {len(violations)} violations from {len(candidate_rules)} candidate rules")
    else:
        # Fallback: dùng tất cả rules nếu không có candidate từ embedding
        violations = forward_chaining(current_facts)
        print(f"[FC] No candidate rules from embedding, using all {len(RULES)} rules")

    # BƯỚC 3: KIỂM TRA RULES ĐÃ MATCH ĐẦY ĐỦ CHƯA
    fully_matched = []
    for rule in violations:
        conds = rule["conditions"]
        all_matched = True
        for slot, required in conds.items():
            if slot not in current_facts or not any(req in current_facts[slot] for req in required):
                all_matched = False
                break
        if all_matched:
            fully_matched.append(rule)

    # BƯỚC 4: NẾU CÓ RULES MATCH ĐẦY ĐỦ → TRẢ KẾT QUẢ
    if fully_matched:
        results = []
        for v in fully_matched:
            penalty_info = v.get("penalty", {})
            results.append({
                "id": v.get("code") or v["id"],
                "title": v.get("title", "Vi phạm giao thông"),
                "legal_ref": penalty_info.get("legal_ref", "Nghị định 168/2024"),
                "penalty": f"{penalty_info.get('amount', '0')} {penalty_info.get('unit', 'đồng')}",
                "additional": penalty_info.get("additional", ""),
                "description": v.get("conclusion", "Bạn đã vi phạm luật giao thông đường bộ")
            })

        # Vẫn có thể hỏi thêm cho các rules khác
        missing = find_missing_conditions(current_facts, req.session_id, candidate_rules)
        questions = []
        if missing:
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

        return QueryResponse(
            status="result",
            message=f"Đã phát hiện {len(fully_matched)} hành vi vi phạm!",
            violations=results,
            questions=questions,
            session_facts=current_facts,
            candidate_rules_count=len(candidate_rules)
        )

    # BƯỚC 5: SUY DIỄN LÙI - HỎI THÊM THÔNG TIN
    missing = find_missing_conditions(current_facts, req.session_id, candidate_rules)
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
        
        return QueryResponse(
            status="need_info",
            message="Mình cần thêm thông tin để tư vấn chính xác!",
            questions=questions,
            session_facts=current_facts,
            candidate_rules_count=len(candidate_rules)
        )

    # BƯỚC 6: KHÔNG TÌM THẤY VI PHẠM
    return QueryResponse(
        status="unknown",
        message="Mình chưa hiểu bạn vi phạm lỗi gì. Hãy kể rõ hơn nhé!\nVí dụ: 'Tôi vượt đèn đỏ bằng xe máy, không đội mũ bảo hiểm'",
        session_facts=current_facts,
        candidate_rules_count=len(candidate_rules)
    )

# ============== UTILS ==============
@app.post("/reset/{session_id}")
def reset(session_id: str):
    r.delete(f"session:{session_id}")
    r.delete(f"asked:{session_id}")
    return {"status": "reset"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rules_count": len(RULES),
        "conditions_count": len(RULE_CONDITIONS),
        "redis": r.ping(),
        "model_loaded": model is not None
    }

@app.get("/debug/embedding")
def debug_embedding(text: str = "vượt đèn đỏ"):
    embedding = get_embedding(text)
    return {
        "text": text,
        "embedding_length": len(embedding),
        "embedding_sample": embedding[:5] if embedding else None
    }
