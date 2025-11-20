# file: app.py (PHIÊN BẢN TỐI ƯU - LOẠI BỎ FORWARD CHAINING)
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

app = FastAPI(title="Vietnam Traffic Law Inference Engine", version="4.0.0")

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
def standardize_attribute(attr: str) -> str:
    """Chuẩn hóa tên attribute"""
    mapping = {
        'vehicle': 'vehicle_type',  # Map 'vehicle' -> 'vehicle_type'
        'xe_may': 'vehicle_type',
        'cho_nguoi_khong_mu': 'action',
        'mo_ta_hanh_vi': 'action'
    }
    return mapping.get(attr, attr)

def standardize_fact_attributes(facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Chuẩn hóa tất cả attribute names trong facts"""
    standardized = {}
    
    for attr, values in facts.items():
        new_attr = standardize_attribute(attr)
        standardized.setdefault(new_attr, []).extend(values)
    
    return standardized

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

            # Load conditions với attribute đã chuẩn hóa
            cond_q = select(cond_table).where(cond_table.c.rule_id == rule_id)
            conds = {}
            for crow in db.execute(cond_q).fetchall():
                original_attr = crow.attribute
                attr = standardize_attribute(original_attr)  # CHUẨN HÓA ATTRIBUTE
                val = crow.value
                embedding = crow.embedding
                
                print(f"[LOAD_COND] Rule {rule_id}: original_attr={original_attr} -> standardized_attr={attr}, val={val!r}")
                
                # Xử lý embedding (giữ nguyên)
                embedding_vec = None
                try:
                    if embedding:
                        if isinstance(embedding, (list, np.ndarray)):
                            embedding_vec = list(embedding) if isinstance(embedding, np.ndarray) else embedding
                        elif isinstance(embedding, str):
                            embedding_vec = json.loads(embedding)
                        else:
                            text_to_embed = f"{attr} {val}"
                            embedding_vec = get_embedding(text_to_embed)
                    else:
                        text_to_embed = f"{attr} {val}"
                        embedding_vec = get_embedding(text_to_embed)
                except Exception as e:
                    print(f"[LOAD_COND] ERROR processing embedding: {e}")
                    text_to_embed = f"{attr} {val}"
                    embedding_vec = get_embedding(text_to_embed)
                
                if isinstance(embedding_vec, np.ndarray):
                    embedding_vec = embedding_vec.tolist()
                
                # Store condition với attribute đã chuẩn hóa
                rule_cond = {
                    "rule_id": rule_id,
                    "attribute": attr,  # Dùng attribute đã chuẩn hóa
                    "original_attribute": original_attr,  # Giữ lại để debug
                    "value": val,
                    "embedding": embedding_vec
                }
                rule_conditions.append(rule_cond)
                
                # Parse values cho traditional matching
                try:
                    values = json.loads(val) if val and val.strip().startswith('[') else [v.strip() for v in val.split(',') if v.strip()]
                except Exception as e:
                    print(f"[LOAD_COND] ERROR parsing values: {e}")
                    values = [val.strip()] if val else []
                
                values = [v.lower() for v in values]
                conds.setdefault(attr, []).extend(values)  # Dùng attribute đã chuẩn hóa
            
            rule["conditions"] = conds
            rule["priority"] = int(rule.get("priority", 0))

            # Load penalty (giữ nguyên)
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
                        "legal_ref": penalty.get("law_ref", ""),
                        "article": penalty.get("article", "")
                    }

            rules.append(rule)
        
        # Sort rules by priority
        rules.sort(key=lambda x: x["priority"], reverse=True)
        
        print(f"\nLoaded {len(rules)} rules and {len(rule_conditions)} conditions with embeddings")
        print(f"Sample attributes in rules: {list(set([cond['attribute'] for cond in rule_conditions]))}")
        
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
        "buông tay": ["buông tay", "buông cả hai tay", "không cầm tay lái", "thả tay lái", "bỏ tay lái"],
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

def build_unified_attributes_mapping():
    """Xây dựng mapping thống nhất giữa các attribute names"""
    unified_map = {}
    
    # Map từ rule_conditions
    for rule_cond in RULE_CONDITIONS:
        attr = rule_cond["attribute"]
        value = rule_cond["value"]
        
        # Chuẩn hóa giá trị
        try:
            values = json.loads(value) if value and value.strip().startswith('[') else [v.strip() for v in value.split(',') if v.strip()]
        except:
            values = [value.strip()] if value else []
        
        values = [v.lower() for v in values]
        
        for v in values:
            unified_map[v] = attr
    
    # Map từ VIETNAMESE_KEYWORDS để hỗ trợ extract_facts_smart
    for slot, groups in VIETNAMESE_KEYWORDS.items():
        for canonical, phrases in groups.items():
            for p in phrases:
                unified_map[p.lower()] = slot
    
    return unified_map

def extract_facts_smart(text: str) -> Dict[str, List[str]]:
    text = text.lower().strip()
    facts = {}
    
    print(f"[EXTRACT] Processing text: '{text}'")
    
    # Xây dựng mapping thống nhất
    ATTRIBUTES_MAPPING = build_unified_attributes_mapping()
    
    # Strategy 1: Tìm các từ khóa trong VIETNAMESE_KEYWORDS (ưu tiên cao)
    for slot, groups in VIETNAMESE_KEYWORDS.items():
        for canonical, phrases in groups.items():
            for phrase in phrases:
                if phrase in text:
                    facts.setdefault(slot, []).append(canonical)
                    print(f"[EXTRACT] Found VIETNAMESE_KEYWORD: '{phrase}' -> {slot}:{canonical}")
    
    # Strategy 2: Tìm các giá trị từ rule_conditions (cẩn thận hơn)
    for value, attr in ATTRIBUTES_MAPPING.items():
        # LOẠI BỎ CÁC TỪ NOISE
        if (not value or len(value.strip()) <= 2 or 
            value in ['máy', 'tay', 'xe', 'có', 'không', 'và']):
            continue
            
        # Kiểm tra nếu giá trị có trong text VÀ có ý nghĩa
        if value in text and is_meaningful_value(value):
            standardized_attr = standardize_attribute(attr)
            # Tránh trùng lặp
            if value not in facts.get(standardized_attr, []):
                facts.setdefault(standardized_attr, []).append(value)
                print(f"[EXTRACT] Found rule_condition: '{value}' -> {standardized_attr}")

    # Thêm các từ khóa đặc biệt
    if 'buông' in text and 'tay' in text:
        if 'buông tay' not in facts.get('action', []):
            facts.setdefault('action', []).append('buông tay')
            print(f"[EXTRACT] Added special case: buông tay -> action")
    
    # Chuẩn hóa các attribute names
    facts = standardize_fact_attributes(facts)
    
    # Dedup
    for k in facts:
        facts[k] = list(set(facts[k]))
    
    print(f"[EXTRACT] Final facts: {facts}")
    return facts

def is_meaningful_value(value: str) -> bool:
    """Kiểm tra xem giá trị có ý nghĩa không (tránh các từ noise)"""
    value_lower = value.lower().strip()
    
    # Danh sách các từ không có ý nghĩa khi đứng một mình
    noise_words = {
        'máy', 'tay', 'xe', 'có', 'không', 'và', 'hoặc', 'trên', 'dưới',
        'trong', 'ngoài', 'khi', 'lúc', 'bằng', 'với', 'cho', 'từ'
    }
    
    if value_lower in noise_words:
        return False
    
    # Các từ phải có độ dài tối thiểu hoặc nằm trong danh sách từ có nghĩa
    meaningful_short_words = {
        'vượt đèn đỏ', 'buông tay', 'xe máy', 'xe ô tô', 'không đội mũ'
    }
    
    if value_lower in meaningful_short_words:
        return True
    
    return len(value_lower) >= 3
# ============== INFERENCE ENGINE MỚI ==============
def check_fully_matched_rules(facts: Dict[str, List[str]], candidate_rules: List[Dict]) -> List[Dict]:
    """Kiểm tra rules nào đã đủ TẤT CẢ điều kiện - ƯU TIÊN RULES TRÙNG NHIỀU FACTS"""
    fully_matched = []
    
    print(f"[FULL_MATCH] Strict checking {len(candidate_rules)} rules with facts: {facts}")
    
    for rule in candidate_rules:
        conds = rule.get("conditions", {})
        all_conditions_met = True
        
        print(f"[FULL_MATCH] Checking rule {rule['id']} - requires: {list(conds.keys())}")
        
        # Kiểm tra TẤT CẢ các điều kiện của rule
        for slot, required_values in conds.items():
            if slot not in facts:
                print(f"[FULL_MATCH]   ❌ Missing slot '{slot}' in facts")
                all_conditions_met = False
                break
            
            # Tìm xem có giá trị fact nào khớp với required value không
            condition_met = False
            for req in required_values:
                for fact_value in facts[slot]:
                    if is_exact_condition_match(fact_value, req):
                        print(f"[FULL_MATCH]   ✓ '{fact_value}' matches '{req}'")
                        condition_met = True
                        break
                if condition_met:
                    break
            
            if not condition_met:
                print(f"[FULL_MATCH]   ❌ No match for {slot}. Facts: {facts[slot]}, Required: {required_values}")
                all_conditions_met = False
                break
        
        if all_conditions_met:
            # TÍNH ĐIỂM TRÙNG KHỚP: số attribute của rule có trong facts
            matched_attributes = set(conds.keys()).intersection(set(facts.keys()))
            match_score = len(matched_attributes)
            
            print(f"[FULL_MATCH] ✅ Rule {rule['id']} PASSED - matched {match_score} attributes: {matched_attributes}")
            fully_matched.append({
                "rule": rule,
                "match_score": match_score,
                "matched_attributes": matched_attributes
            })
        else:
            print(f"[FULL_MATCH] ❌ Rule {rule['id']} FAILED - missing some conditions")
    
    # ƯU TIÊN: Sắp xếp rules theo số attribute trùng với facts (cao nhất trước)
    if fully_matched:
        fully_matched.sort(key=lambda x: x["match_score"], reverse=True)
        
        # SỬA LỖI F-STRING: Tách thành biến riêng
        sorted_rules_info = []
        for r in fully_matched:
            rule_id = r['rule']['id']
            score = r['match_score']
            sorted_rules_info.append(f"Rule {rule_id} (score: {score})")
        
        print(f"[FULL_MATCH] Sorted by match score: {sorted_rules_info}")
        
        # Chỉ trả về rules (bỏ qua metadata)
        fully_matched_rules = [item["rule"] for item in fully_matched]
    else:
        fully_matched_rules = []
    
    print(f"[FULL_MATCH] Found {len(fully_matched_rules)} fully matched rules")
    return fully_matched_rules
def is_exact_match_for_rule(fact_value: str, required_value: str) -> bool:
    """Kiểm tra match chính xác cho rule conditions"""
    fact_lower = fact_value.lower().strip()
    req_lower = required_value.lower().strip()
    
    # Match chính xác (hoàn toàn giống nhau)
    if fact_lower == req_lower:
        return True
    
    # Match các giá trị đặc biệt
    special_matches = {
        'xe máy': ['xe_may', 'xe máy', 'xe may', 'mô tô', 'xe gắn máy'],
        'buông tay': ['buông tay', 'buông cả hai tay', 'buông tay lái'],
        'vượt đèn đỏ': ['vượt đèn đỏ', 'băng đèn đỏ'],
        'không đội mũ bảo hiểm': ['không đội mũ', 'không mũ bảo hiểm']
    }
    
    for canonical, variants in special_matches.items():
        if fact_lower in variants and req_lower in variants:
            return True
    
    # Match khi fact_value là một phần của required_value (chỉ cho các cụm từ đặc biệt)
    meaningful_phrases = {
        'buông tay', 'xe máy', 'vượt đèn đỏ', 'không đội mũ', 
        'ngược chiều', 'lạng lách', 'chở quá người'
    }
    
    if fact_lower in meaningful_phrases and fact_lower in req_lower:
        return True
        
    if req_lower in meaningful_phrases and req_lower in fact_lower:
        return True
    
    return False
def is_exact_condition_match(fact_value: str, required_value: str) -> bool:
    """Kiểm tra match chính xác giữa fact và required value"""
    fact_lower = fact_value.lower().strip()
    req_lower = required_value.lower().strip()
    
    # 1. Match chính xác
    if fact_lower == req_lower:
        return True
    
    # 2. Mapping các giá trị tương đương
    value_equivalence = {
        # Action mappings
        'buông tay': ['buông cả hai tay', 'buông tay lái'],
        'vượt đèn đỏ': ['băng đèn đỏ', 'vượt đèn'],
        'không đội mũ bảo hiểm': ['không đội mũ', 'không mũ bảo hiểm'],
        
        # Vehicle type mappings  
        'xe máy': ['xe_may', 'xe may', 'mô tô', 'xe gắn máy'],
        'xe ô tô': ['oto', 'xe hơi'],
        'xe đạp': ['xe đạp điện']
    }
    
    # Kiểm tra trong mapping
    for canonical, variants in value_equivalence.items():
        all_variants = [canonical] + variants
        if fact_lower in all_variants and req_lower in all_variants:
            return True
    
    # 3. Match một phần chỉ cho các cụm từ đặc biệt
    special_phrases = {
        'buông tay', 'xe máy', 'vượt đèn đỏ', 'không đội mũ',
        'ngược chiều', 'lạng lách', 'chở quá người'
    }
    
    # Chỉ match nếu cả hai đều thuộc cùng một nhóm ý nghĩa
    for phrase in special_phrases:
        if phrase in fact_lower and phrase in req_lower:
            return True
    
    return False

def find_partially_matched_rules(facts: Dict[str, List[str]], candidate_rules: List[Dict]) -> List[Dict]:
    """Tìm rules có ít nhất 1 điều kiện khớp nhưng CHƯA ĐỦ điều kiện"""
    partially_matched = []
    
    print(f"[PARTIAL_RULES] Finding rules with some but not all conditions matched")
    
    for rule in candidate_rules:
        conds = rule["conditions"]
        matched_slots = []
        missing_slots = []
        
        for slot, required_values in conds.items():
            slot_matched = False
            if slot in facts:
                for req in required_values:
                    for fact_value in facts[slot]:
                        if is_exact_condition_match(fact_value, req):
                            matched_slots.append(slot)
                            slot_matched = True
                            break
                    if slot_matched:
                        break
            
            if not slot_matched:
                missing_slots.append(slot)
        
        # Chỉ thêm rules có ít nhất 1 slot khớp và còn thiếu slot
        if matched_slots and missing_slots:
            match_score = len(matched_slots) / len(conds)
            partially_matched.append({
                "rule": rule,
                "matched_slots": matched_slots,
                "missing_slots": missing_slots,
                "match_score": match_score
            })
            print(f"[PARTIAL_RULES] Rule {rule['id']}: matched={matched_slots}, missing={missing_slots}, score={match_score:.2f}")
    
    # Sắp xếp theo độ khớp cao nhất
    partially_matched.sort(key=lambda x: x["match_score"], reverse=True)
    return partially_matched
def generate_smart_questions(partially_matched_rules: List[Dict], session_id: str) -> List[Dict]:
    """Tạo câu hỏi thông minh từ các rules còn thiếu điều kiện"""
    questions = []
    asked_slots = set()
    
    # Lấy danh sách slot đã hỏi trước đó
    asked_key = f"asked:{session_id}"
    previously_asked = set(r.smembers(asked_key))
    
    for rule_info in partially_matched_rules:
        rule = rule_info["rule"]
        
        for slot in rule_info["missing_slots"]:
            # Skip nếu đã hỏi slot này trong phiên hiện tại hoặc trước đó
            if slot in asked_slots or slot in previously_asked:
                continue
                
            template = SMART_QUESTIONS.get(slot)
            if template:
                questions.append({
                    "slot": slot,
                    "question": template["question"],
                    "options": template["options"],
                    "related_rule_id": rule["id"]
                })
                asked_slots.add(slot)
        
        # Chỉ hỏi tối đa 2 câu mỗi lần
        if len(questions) >= 2:
            break
    
    # Lưu lại các slot sắp hỏi
    if asked_slots:
        r.sadd(asked_key, *asked_slots)
        r.expire(asked_key, REDIS_TTL)
    
    return questions

def find_rules_by_keywords(facts: Dict[str, List[str]]) -> List[Dict]:
    """Tìm candidate rules dựa trên keywords - CHỈ TÌM RULES CÓ LIÊN QUAN"""
    candidate_rules = []
    
    print(f"[KEYWORD_MATCH] Searching rules with facts: {facts}")
    
    if not facts:
        return candidate_rules
    
    # Tạo danh sách các từ khóa quan trọng từ facts
    important_keywords = set()
    for slot, values in facts.items():
        for value in values:
            if len(value) >= 3:  # Chỉ lấy từ có ý nghĩa
                important_keywords.add(value.lower())
    
    print(f"[KEYWORD_MATCH] Important keywords: {important_keywords}")
    
    for rule in RULES:
        conds = rule.get("conditions", {})
        rule_relevance_score = 0
        
        # Tính điểm relevance dựa trên số slot khớp
        for slot in conds.keys():
            if slot in facts:
                rule_relevance_score += 1
        
        # Chỉ xem xét rules có ít nhất 1 slot khớp
        if rule_relevance_score > 0:
            candidate_rules.append(rule)
            print(f"[KEYWORD_MATCH] Added rule {rule['id']} as candidate (relevance score: {rule_relevance_score})")
    
    print(f"[KEYWORD_MATCH] Found {len(candidate_rules)} candidate rules")
    return candidate_rules
def is_exact_match(fact_value: str, required_value: str) -> bool:
    """Kiểm tra match chính xác hơn, tránh match các từ con"""
    fact_lower = fact_value.lower().strip()
    req_lower = required_value.lower().strip()
    
    # Loại bỏ các từ quá ngắn có thể gây match sai
    short_noise_words = {'máy', 'xe', 'tay', 'có', 'không', 'và', 'hoặc', 'trên', 'dưới'}
    
    # Nếu fact_value là từ đơn và nằm trong noise words -> không match
    if fact_lower in short_noise_words:
        return False
    
    # Match chính xác (hoàn toàn giống nhau)
    if fact_lower == req_lower:
        return True
    
    # Match khi fact_value là một phần của required_value VÀ có ý nghĩa
    if fact_lower in req_lower:
        # Chỉ match nếu fact_value đủ dài và không phải là noise
        if len(fact_lower) >= 4 and fact_lower not in short_noise_words:
            # Kiểm tra xem có phải là từ có nghĩa không
            meaningful_words = {
                'buông tay', 'buông cả hai tay', 'xe máy', 'vượt đèn đỏ', 
                'không đội mũ', 'ngược chiều', 'vượt ẩu', 'lạng lách'
            }
            if fact_lower in meaningful_words:
                return True
    
    # Match khi required_value là một phần của fact_value
    if req_lower in fact_lower:
        if len(req_lower) >= 4 and req_lower not in short_noise_words:
            meaningful_required = {
                'buông tay', 'xe máy', 'vượt đèn', 'đội mũ', 'ngược chiều'
            }
            if req_lower in meaningful_required:
                return True
    
    return False
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
    candidate_rules_count: int = 0

# ============== SESSION ==============
def get_facts(sid: str) -> Dict[str, List[str]]:
    data = r.get(f"session:{sid}")
    return json.loads(data) if data else {}

def save_facts(sid: str, facts: Dict[str, List[str]]):
    r.setex(f"session:{sid}", REDIS_TTL, json.dumps(facts))
def find_partially_matched_rules_only(facts: Dict[str, List[str]]) -> List[Dict]:
    """Tìm rules có ÍT NHẤT 1 điều kiện khớp với facts (dùng cho candidate)"""
    candidate_rules = []
    
    print(f"[PARTIAL_MATCH] Finding rules with at least 1 matching condition for: {facts}")
    
    if not facts:
        return candidate_rules
    
    for rule in RULES:
        conds = rule.get("conditions", {})
        has_match = False
        
        print(f"[PARTIAL_MATCH] Checking rule {rule['id']}: conditions={list(conds.keys())}")
        
        for slot, required_values in conds.items():
            if slot in facts:
                print(f"[PARTIAL_MATCH]   Slot '{slot}' found in facts: {facts[slot]}")
                print(f"[PARTIAL_MATCH]   Required values: {required_values}")
                
                for req in required_values:
                    for fact_value in facts[slot]:
                        if is_exact_condition_match(fact_value, req):
                            if rule not in candidate_rules:  # Tránh trùng lặp
                                candidate_rules.append(rule)
                                has_match = True
                                print(f"[PARTIAL_MATCH]   ✓ Rule {rule['id']} added - matched: {slot}='{fact_value}' with '{req}'")
                            break
                    if has_match:
                        break
            if has_match:
                break
    
    print(f"[PARTIAL_MATCH] Found {len(candidate_rules)} candidate rules")
    return candidate_rules
# ============== MAIN ENDPOINT TỐI ƯU ==============
# ============== FIXED MAIN ENDPOINT ==============
@app.post("/infer", response_model=QueryResponse)
async def infer(req: QueryRequest):
    if not req.session_id:
        raise HTTPException(400, "session_id required")
    
    print(f"[INFER] session={req.session_id} message={req.message} text={req.text}")
    
    current_facts = get_facts(req.session_id)
    new_facts = {}
    candidate_rules = []

    # BƯỚC 1: EXTRACT FACTS TỪ INPUT
    input_text = req.get_input_text()
    if input_text:
        extracted_facts = extract_facts_smart(input_text)
        for k, v in extracted_facts.items():
            new_facts.setdefault(k, []).extend(v)

    # Merge facts
    if req.facts:
        for k, v in req.facts.items():
            new_facts.setdefault(k, []).extend(v)

    for k, v in new_facts.items():
        current_facts.setdefault(k, []).extend(v)
        current_facts[k] = list(set(current_facts[k]))

    print(f"[INFER] Current facts: {current_facts}")
    
    # BƯỚC 2: TÌM CANDIDATE RULES (chỉ cần có 1 điều kiện khớp)
    if input_text:
        query_embedding = get_embedding(input_text)
        similar_rules_info = find_similar_rules(query_embedding, threshold=0.7)
        candidate_rules = [info["rule"] for info in similar_rules_info]

    # Fallback: keyword matching
    if not candidate_rules:
        candidate_rules = find_partially_matched_rules_only(current_facts)
    
    save_facts(req.session_id, current_facts)

    # BƯỚC 3: KIỂM TRA RULES ĐÃ ĐỦ ĐIỀU KIỆN CHƯA
    fully_matched = check_fully_matched_rules(current_facts, candidate_rules)
    
    # NẾU CÓ RULES ĐỦ ĐIỀU KIỆN → TRẢ KẾT QUẢ
    if fully_matched:
        print(f"[INFER] Found {len(fully_matched)} fully matched rules")
        
        # CHỌN RULE TỐT NHẤT: Ưu tiên rules có nhiều attribute trùng với facts nhất
        best_rule = fully_matched[0]  # Đã được sắp xếp theo match_score
        print(f"[INFER] Selected best rule: {best_rule['id']} (highest match score)")
        
        results = []
        penalty_info = best_rule.get("penalty", {})
        
        # THÊM THÔNG TIN ĐIỀU KHOẢN VÀ CĂN CỨ
        result_item = {
            "id": best_rule.get("code") or best_rule["id"],
            "title": best_rule.get("title", "Vi phạm giao thông"),
            "legal_ref": penalty_info.get("legal_ref", "Nghị định 168/2024"),
            "penalty": f"{penalty_info.get('amount', '0')} {penalty_info.get('unit', 'đồng')}",
            "additional": penalty_info.get("additional", ""),
            "article": penalty_info.get('article', ''),
            "description": best_rule.get("conclusion", "Bạn đã vi phạm luật giao thông đường bộ")

        }
        
            
        results.append(result_item)

        return QueryResponse(
            status="result",
            message=f"Đã xác định hành vi vi phạm!",
            violations=results,
            session_facts=current_facts,
            candidate_rules_count=len(candidate_rules)
        )

    # BƯỚC 4: NẾU CHƯA ĐỦ → TÌM RULES CÓ THỂ MATCH MỘT PHẦN VÀ HỎI THÊM
    partially_matched = find_partially_matched_rules(current_facts, candidate_rules)
    if partially_matched:
        questions = generate_smart_questions(partially_matched, req.session_id)
        
        if questions:
            return QueryResponse(
                status="need_info",
                message="Mình cần thêm thông tin để tư vấn chính xác!",
                questions=questions,
                session_facts=current_facts,
                candidate_rules_count=len(candidate_rules)
            )

    # BƯỚC 5: KHÔNG TÌM THẤY GÌ
    return QueryResponse(
        status="unknown",
        message="Mình chưa hiểu bạn vi phạm lỗi gì. Hãy kể rõ hơn nhé!",
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

@app.get("/debug/rules")
def debug_rules(rule_id: Optional[int] = None):
    """Debug endpoint để xem thông tin rules"""
    if rule_id:
        rule = next((r for r in RULES if r["id"] == rule_id), None)
        return {"rule": rule}
    
    return {
        "total_rules": len(RULES),
        "rules_sample": RULES[:3] if RULES else []
    }
