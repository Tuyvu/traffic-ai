from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import math
import re
import networkx as nx
from collections import defaultdict

app = FastAPI(title="Inference API", version="1.0")

# -----------------------------
# 🧩 Cấu trúc dữ liệu đầu vào
# -----------------------------
class Rule(BaseModel):
    id: int
    input: str
    output: str
    formula: str
    converted_formula: Optional[str] = None

class InferenceRequest(BaseModel):
    rules: List[Rule]
    event: str      # ví dụ: "a=3,b=4,C=60" hoặc "a,b,C" cho symbolic
    conclusion: str # ví dụ: "c"
    type: str = "forward"  # "forward" hoặc "backward"
    graph_type: str = "fpg"  # "fpg" hoặc "rpg"

# -----------------------------
# Hàm hỗ trợ
# -----------------------------
def convert_formula(formula: str) -> str:
    if '=' in formula:
        lhs, rhs = formula.split('=', 1)
        rhs = rhs.strip()
    else:
        rhs = formula
    rhs = rhs.replace('²', '**2')
    rhs = rhs.replace('·', '*')
    rhs = rhs.replace('√', 'math.sqrt')
    def replace_trig(m):
        func = m.group(1)
        arg = m.group(2)
        if func in ['acos', 'asin', 'atan']:
            return f'math.degrees(math.{func}({arg}))'
        else:
            return f'math.{func}(math.radians({arg}))'
    rhs = re.sub(r'(cos|sin|tan|acos|asin|atan)([A-Z])', replace_trig, rhs)
    return rhs

def parse_event(event_str: str) -> Dict[str, float]:
    """Chuyển chuỗi a=3,b=4,C=60 -> dict {'a':3,'b':4,'C':60}, hoặc a,b,C -> {'a':None, ...}"""
    values = {}
    for part in event_str.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=")
            k = k.strip()
            try:
                v = float(v.strip())
            except:
                v = None
        else:
            k = part
            v = None
        if k:
            values[k] = v
    return values

def build_graph(rules: List[Rule], graph_type: str) -> Dict[str, List[Dict]]:
    nodes = []
    edges = []

    if graph_type == "fpg":
        # =============================
        # FPG: Facts Precedence Graph (KHÔNG TRÙNG LẶP)
        # =============================
        seen_facts = set()
        seen_edges = set()  # ← THÊM NÀY: loại bỏ trùng lặp

        # Bước 1: Thu thập TẤT CẢ facts trước
        all_facts = set()
        for rule in rules:
            premises = [p.strip() for p in rule.input.split(",") if p.strip()]
            output = rule.output.strip()
            all_facts.update(premises)
            all_facts.add(output)

        # Bước 2: Tạo node cho facts (1 lần duy nhất)
        for fact in all_facts:
            nodes.append({"data": {"id": fact, "label": fact}})

        # Bước 3: Tạo edges KHÔNG TRÙNG LẶP
        for rule in rules:
            premises = [p.strip() for p in rule.input.split(",") if p.strip()]
            output = rule.output.strip()
            
            for premise in premises:
                # Tạo unique edge ID: "source-target"
                edge_key = f"{premise}→{output}"
                
                if edge_key not in seen_edges:
                    edges.append({
                        "data": {
                            "source": premise,
                            "target": output,
                        }
                    })
                    seen_edges.add(edge_key)

    elif graph_type == "rpg":
        # RPG: Giữ nguyên (đã đúng)
        seen_rules = set()
        for rule in rules:
            rule_id = f"R{rule.id}"
            if rule_id not in seen_rules:
                nodes.append({
                    "data": {
                        "id": rule_id,
                        "label": f"R{rule.id}"
                    }
                })
                seen_rules.add(rule_id)

        # RPG edges (không trùng lặp)
        output_to_rules = {}
        for rule in rules:
            out = rule.output.strip()
            if out not in output_to_rules:
                output_to_rules[out] = []
            output_to_rules[out].append(rule.id)

        seen_rpg_edges = set()
        for rule_j in rules:
            rj_id = f"R{rule_j.id}"
            premises = [p.strip() for p in rule_j.input.split(",") if p.strip()]
            for premise in premises:
                if premise in output_to_rules:
                    for ri_id_num in output_to_rules[premise]:
                        ri_id = f"R{ri_id_num}"
                        if ri_id != rj_id:
                            edge_key = f"{ri_id}→{rj_id}"
                            if edge_key not in seen_rpg_edges:
                                edges.append({
                                    "data": {
                                        "source": ri_id,
                                        "target": rj_id,
                                    }
                                })
                                seen_rpg_edges.add(edge_key)

    return {"nodes": nodes, "edges": edges}
def get_shortest_path_rules(rules: List[Rule], known: set, goal: str, graph_type: str) -> List[Rule]:
    """
    Select rules based on shortest path to goal in FPG (for forward) or RPG (for backward).
    """
    G = nx.DiGraph()
    
    if graph_type == 'fpg':
        # Build FPG for shortest path
        for rule in rules:
            premises = [p.strip() for p in rule.input.split(",") if p.strip()]
            output = rule.output.strip()
            for p in premises:
                G.add_edge(p, output, rule_id=rule.id)
    
        # Find rules on shortest paths from known facts to goal
        applicable_rules = []
        for fact in known:
            try:
                paths = nx.all_shortest_paths(G, source=fact, target=goal)
                for path in paths:
                    for i in range(len(path) - 1):
                        rule_id = G[path[i]][path[i+1]].get('rule_id')
                        if rule_id is not None:
                            rule = next(r for r in rules if r.id == rule_id)
                            if rule not in applicable_rules:
                                applicable_rules.append(rule)
            except nx.NetworkXNoPath:
                continue
        return applicable_rules if applicable_rules else rules  # Fallback to all rules if no path
    
    elif graph_type == 'rpg':
        # Build RPG for shortest path
        rule_outputs = {rule.id: rule.output.strip() for rule in rules}
        for ri in rules:
            ri_output = ri.output.strip()
            for rj in rules:
                if ri.id != rj.id:
                    rj_premises = [p.strip() for p in rj.input.split(",") if p.strip()]
                    if ri_output in rj_premises:
                        G.add_edge(f"R{ri.id}", f"R{rj.id}")
        
        # Find rules leading to goal-producing rules
        goal_rules = [r for r in rules if r.output.strip() == goal]
        applicable_rules = []
        for gr in goal_rules:
            try:
                for rule in rules:
                    if nx.has_path(G, f"R{rule.id}", f"R{gr.id}") or rule.id == gr.id:
                        if rule not in applicable_rules:
                            applicable_rules.append(rule)
            except nx.NodeNotFound:
                continue
        return applicable_rules if applicable_rules else rules  # Fallback to all rules if no path
    
    else:
        return rules  # Fallback to all rules if graph_type is invalid

# -----------------------------
# Hàm xử lý suy diễn
# -----------------------------
def forward_chain(rules: List[Rule], known: set, goal: str, values: Dict[str, float]):
    derived = set(known)
    steps = []
    used_rules = set()

    applicable_rules = rules
    changed = True

    while changed:
        changed = False
        for rule in applicable_rules:
            premises = [p.strip() for p in rule.input.split(",") if p.strip()]
            conclusion = rule.output.strip()
            expr = rule.converted_formula.strip()

            if all(p in derived for p in premises) and conclusion not in derived:
                derived.add(conclusion)
                changed = True
                used_rules.add(rule.id)

                step = {
                    "rule_id": rule.id,
                    "premises": premises,
                    "conclusion": conclusion,
                    "formula": rule.formula,
                    "converted_formula": expr,
                    "result": None
                }

                if all(values.get(p) is not None for p in premises):
                    try:
                        result = eval(expr, {
                            "__builtins__": None,
                            "math": math,
                            "sqrt": math.sqrt,
                            "cos": math.cos,
                            "sin": math.sin,
                            "radians": math.radians,
                            "degrees": math.degrees
                        }, values)
                        values[conclusion] = result
                        step["result"] = result
                    except Exception as e:
                        step["result"] = f"Lỗi: {e}"
                else:
                    step["result"] = "Derived symbolically"

                steps.append(step)

                # Nếu đạt goal → trả về ngay
                if conclusion == goal:
                    return {
                        "success": True,
                        "conclusion": f"{goal} = {values.get(goal)}",
                        "trace": steps,
                        "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules)]
                    }

    # === CHỈ 1 LẦN RETURN CUỐI CÙNG ===
    success = goal in derived
    conclusion_msg = f"{goal} = {values.get(goal, 'Không tìm thấy')}"
    return {
        "success": success,
        "conclusion": conclusion_msg,
        "trace": steps,
        "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules)]
    }  # SAU KHI HẾT VÒNG LẶP
    success = goal in derived
    conclusion_msg = f"{goal} = {values.get(goal, 'Không tìm thấy')}"
    return {
        "success": success,
        "conclusion": conclusion_msg,
        "trace": steps,
        "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules)]
    }
    # Kiểm tra cuối cùng
    success = goal in derived
    conclusion_msg = f"{goal} = {values.get(goal, 'Không tìm thấy')}"
    return {
        "success": success,
        "conclusion": conclusion_msg,
        "trace": steps,
        "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules)]
    }
    success = goal in derived
    conclusion_msg = f"{goal} = {values.get(goal, 'Derived symbolically' if success else 'Không tìm thấy')}"
    return {
        "success": success,
        "conclusion": conclusion_msg,
        "trace": steps,
        "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules)]
    }

def backward_chain(rules: List[Rule], known: set, goal: str, values: Dict[str, float], visited: set = None, steps: List = None):
    if visited is None:
        visited = set()
    if steps is None:
        steps = []
    used_rules = set() if not steps else {s["rule_id"] for s in steps}

    if goal in known:
        return True, steps, used_rules
    if goal in visited:
        return False, steps, used_rules
    visited.add(goal)

    # === SỬA: CONVERT CÔNG THỨC TRƯỚC KHI DÙNG ===
    for rule in rules:
        if rule.converted_formula is None:
            rule.converted_formula = convert_formula(rule.formula)

    applicable_rules = get_shortest_path_rules(rules, known, goal, 'rpg')

    for rule in applicable_rules:
        if rule.output.strip() == goal:
            premises = [p.strip() for p in rule.input.split(",") if p.strip()]
            all_known = True
            sub_steps = []
            sub_used_rules = set()

            for p in premises:
                if p not in known:
                    success, new_steps, new_used_rules = backward_chain(
                        rules, known, p, values, visited, sub_steps
                    )
                    sub_steps = new_steps
                    sub_used_rules.update(new_used_rules)
                    if not success:
                        all_known = False
                        break

            if all_known:
                step = {
                    "rule_id": rule.id,
                    "premises": premises,
                    "conclusion": goal,
                    "formula": rule.formula,
                    "converted_formula": rule.converted_formula,
                    "result": None
                }

                # === SỬA: DÙNG converted_formula ĐÃ CHUYỂN ĐỘ → RADIAN ===
                if all(values.get(p) is not None for p in premises):
                    try:
                        result = eval(rule.converted_formula, {
                            "__builtins__": None,
                            "math": math,
                            "sqrt": math.sqrt,
                            "cos": math.cos,
                            "sin": math.sin,
                            "radians": math.radians,
                            "degrees": math.degrees
                        }, values)
                        values[goal] = result
                        step["result"] = result
                    except Exception as e:
                        step["result"] = f"Lỗi: {e}"
                else:
                    step["result"] = "Derived symbolically"

                steps.extend(sub_steps)
                steps.append(step)
                used_rules.add(rule.id)
                used_rules.update(sub_used_rules)
                known.add(goal)
                return True, steps, used_rules

    return False, steps, used_rules
# -----------------------------
# ⚡ API Endpoint
# -----------------------------
@app.post("/infer")
async def infer(request: InferenceRequest):
    # === 1. LẤY DỮ LIỆU TỪ REQUEST ===
    rules = request.rules
    event = request.event
    goal = request.conclusion.strip()  # kết luận cần suy ra
    inf_type = request.type  # "forward" hoặc "backward"
    graph_type = request.graph_type  # "fpg" hoặc "rpg"

    # === 2. CHUẨN BỊ DỮ LIỆU ===
    # Convert công thức nếu chưa có
    for rule in rules:
        if rule.converted_formula is None:
            rule.converted_formula = convert_formula(rule.formula)

    # Parse event → known values + known facts
    values = parse_event(event)
    known = set(values.keys())

    # === 3. CHẠY SUY DIỄN ===
    if inf_type == "forward":
        result = forward_chain(rules, known, goal, values)
    elif inf_type == "backward":
        success, trace, used_rules_set = backward_chain(rules, known, goal, values)
        conclusion_msg = f"{goal} = {values.get(goal, 'Derived symbolically' if success else 'Không tìm thấy')}"
        result = {
            "success": success,
            "conclusion": conclusion_msg,
            "trace": trace,
            "used_rules": [f"R{rule_id}" for rule_id in sorted(used_rules_set)]
        }
    else:
        return {"error": "Invalid type: must be 'forward' or 'backward'"}

    # === 4. LỌC TRACE THEO ĐƯỜNG ĐI ĐẾN GOAL ===
    relevant_steps = []
    needed = {goal}
    seen_conclusions = set()

    while needed:
        new_needed = set()
        for step in reversed(result.get("trace", [])):
            conc = step["conclusion"]
            if conc in needed and conc not in seen_conclusions:
                relevant_steps.append(step)
                seen_conclusions.add(conc)
                new_needed.update([p.strip() for p in step["premises"]])
        needed = new_needed - seen_conclusions
        if not needed:
            break

    relevant_steps = relevant_steps[::-1]  # đảo lại thứ tự đúng

    # === 5. TẠO SOLUTION_STEPS ===
    steps_text = []
    for step in relevant_steps:
        formula = step["formula"]
        result_val = step["result"]
        if isinstance(result_val, float):
            result_val = round(result_val, 8)
        elif result_val is None:
            result_val = "Derived symbolically"
        steps_text.append(f"{formula}")
        steps_text.append(f"   {step['conclusion']} = {result_val}\n")

    result["solution_steps"] = "\n".join(steps_text)

    # === 6. CẬP NHẬT KẾT LUẬN ===
    if result["success"]:
        final_val = values.get(goal)
        if isinstance(final_val, float):
            final_val = round(final_val, 8)
        result["conclusion"] = f"{goal} = {final_val}"

    # === 7. ĐỒ THỊ ===
    result["graph"] = build_graph(rules, graph_type)

    return result