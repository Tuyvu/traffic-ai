document.addEventListener("DOMContentLoaded", () => {

    // ===============================
    // 🔹 1. Khai báo biến
    // ===============================
    let rules = [];
    let facts = [];
    let ruleSets = JSON.parse(localStorage.getItem("ruleSets") || "{}");
    let currentSet = null;

    // DOM
    const rulesList = document.getElementById("rules-list");
    const factsList = document.getElementById("facts-list");
    const rulesStorage = document.getElementById("rules-storage");
    const factsStorage = document.getElementById("facts-storage");
    const ruleSetSelector = document.getElementById("rule-set-selector");
    const goalInput = document.querySelector("input[name='goal']");
    const saveBtn = document.getElementById("save-button");

    // ===============================
    // 🔹 2. Cập nhật hidden input
    // ===============================
    function updateHiddenInputs() {
        rulesStorage.value = JSON.stringify(rules);
        factsStorage.value = JSON.stringify(facts);
    }

    // ===============================
    // 🔹 3. Render danh sách luật & giả thiết
    // ===============================
    function renderRules() {
        rulesList.innerHTML = rules.map((r, i) =>
            `<div class="list-item"><span>${r}</span><button type="button" class="btn-delete" data-i="${i}">X</button></div>`
        ).join("");
    }

    function renderFacts() {
        factsList.innerHTML = facts.map((f, i) =>
            `<div class="list-item"><span>${f}</span><button type="button" class="btn-delete" data-i="${i}">X</button></div>`
        ).join("");
    }

    // ===============================
    // 🔹 4. Thêm / Xoá luật & giả thiết
    // ===============================
    document.getElementById("add-fact-button").addEventListener("click", () => {
    const val = document.getElementById("fact-input").value.trim();
    if (!val) return;

    // ✨ Tách theo khoảng trắng hoặc dấu phẩy
    const items = val.split(/[\s,]+/).filter(Boolean);

    facts.push(...items);
    renderFacts();
    updateHiddenInputs();
});


    rulesList.addEventListener("click", e => {
        if (e.target.classList.contains("btn-delete")) {
            rules.splice(e.target.dataset.i, 1);
            renderRules(); updateHiddenInputs();
        }
    });

    factsList.addEventListener("click", e => {
        if (e.target.classList.contains("btn-delete")) {
            facts.splice(e.target.dataset.i, 1);
            renderFacts(); updateHiddenInputs();
        }
    });

    // ===============================
    // 🔹 5. Nhập nhiều luật
    // ===============================
    const bulkForm = document.getElementById("bulk-form");
    document.getElementById("open-bulk-form").onclick = () => bulkForm.classList.remove("hidden");
    document.getElementById("cancel-bulk").onclick = () => bulkForm.classList.add("hidden");

    document.getElementById("confirm-bulk").onclick = () => {
        const raw = document.getElementById("bulk-rules-input").value.trim();
        if (!raw) return alert("⚠️ Vui lòng nhập ít nhất 1 luật!");
        rules.push(...raw.split("\n").map(l => l.replace("->", "→").trim()).filter(Boolean));
        renderRules(); updateHiddenInputs();
        bulkForm.classList.add("hidden");
    };

    // ===============================
    // 🔹 6. Import luật từ file
    // ===============================
    document.getElementById("import-button").onclick = () => document.getElementById("file-importer").click();
    document.getElementById("file-importer").addEventListener("change", async e => {
        const file = e.target.files[0];
        if (!file) return;
        const ext = file.name.split(".").pop().toLowerCase();
        const reader = new FileReader();

        if (ext === "json") {
            reader.onload = e => {
                const data = JSON.parse(e.target.result);
                rules = data.rules || [];
                facts = data.facts || [];
                goalInput.value = data.goal || "";
                renderRules(); renderFacts(); updateHiddenInputs();
            };
            reader.readAsText(file);
        } else {
            reader.onload = e => {
                const lines = e.target.result.split("\n").map(l => l.trim()).filter(l => l);
                lines.forEach(line => {
                    if (line.includes("→") || line.includes("->")) rules.push(line.replace("->", "→"));
                    else if (line.startsWith("GT")) facts.push(...line.match(/\{(.*?)\}/)[1].split(","));
                    else if (line.startsWith("KL")) goalInput.value = line.match(/\{(.*?)\}/)[1];
                });
                renderRules(); renderFacts(); updateHiddenInputs();
            };
            reader.readAsText(file);
        }
    });

    // ===============================
    // 🔹 7. Lưu / Load LocalStorage
    // ===============================
    function saveData() {
        const data = {
            rules, facts,
            goal: goalInput.value,
            inference_type: document.querySelector("input[name='inference_type']:checked").value,
            graph_type: document.querySelector("input[name='graph_type']:checked").value
        };
        localStorage.setItem("inferenceData", JSON.stringify(data));
        alert("💾 Đã lưu dữ liệu vào localStorage!");
    }

    saveBtn.onclick = saveData;
    document.addEventListener("keydown", e => {
        if (e.key === "F11") { e.preventDefault(); saveData(); }
    });

    const saved = localStorage.getItem("inferenceData");
    if (saved) {
        const data = JSON.parse(saved);
        rules = data.rules || [];
        facts = data.facts || [];
        renderRules(); renderFacts();
        if (data.goal) goalInput.value = data.goal;
        updateHiddenInputs();
    }

    // ===============================
    // 🔹 8. Quản lý bộ luật (Rule Sets)
    // ===============================
    function renderRuleSets() {
        ruleSetSelector.innerHTML = `<option value="">-- Chưa chọn bộ luật nào --</option>`;
        for (let k in ruleSets) {
            ruleSetSelector.innerHTML += `<option value="${k}">${k}</option>`;
        }
    }

    renderRuleSets();

    document.getElementById("new-rule-set").onclick = () => document.getElementById("new-rule-set-modal").classList.remove("hidden");
    document.getElementById("cancel-new-set").onclick = () => document.getElementById("new-rule-set-modal").classList.add("hidden");

    document.getElementById("confirm-new-set").onclick = () => {
        const name = document.getElementById("rule-set-name").value.trim();
        const desc = document.getElementById("rule-set-desc").value.trim();
        if (!name) return alert("⚠️ Nhập tên bộ luật!");
        ruleSets[name] = { desc, rules: [], facts: [], goal: "" };
        localStorage.setItem("ruleSets", JSON.stringify(ruleSets));
        renderRuleSets();
        alert(`✅ Đã tạo bộ luật "${name}"`);
        document.getElementById("new-rule-set-modal").classList.add("hidden");
    };

    ruleSetSelector.addEventListener("change", () => {
        const name = ruleSetSelector.value;
        if (!name) return;
        currentSet = name;
        const set = ruleSets[name];
        rules = set.rules || [];
        facts = set.facts || [];
        goalInput.value = set.goal || "";
        renderRules(); renderFacts(); updateHiddenInputs();
    });

    document.getElementById("delete-rule-set").onclick = () => {
        if (!currentSet) return alert("⚠️ Chưa chọn bộ luật!");
        if (confirm(`Xoá bộ luật "${currentSet}"?`)) {
            delete ruleSets[currentSet];
            localStorage.setItem("ruleSets", JSON.stringify(ruleSets));
            renderRuleSets();
            rules = []; facts = [];
            renderRules(); renderFacts();
            updateHiddenInputs();
            alert("🗑 Đã xoá!");
        }
    };
});