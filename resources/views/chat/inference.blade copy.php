<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tư vấn Vi phạm Giao thông</title>
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50">

<div class="min-h-screen p-6" x-data="trafficAdvisor()">
    <div class="max-w-6xl mx-auto grid grid-cols-12 gap-6">

        <!-- Left: Chat -->
        <div class="col-span-7">
            <div class="bg-white rounded-2xl shadow p-4">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-xl font-semibold">Tư vấn vi phạm giao thông</h2>
                </div>

                <div class="border rounded-lg p-3 h-[420px] overflow-y-auto bg-gray-50">
                    <template x-if="conversation.length === 0">
                        <div class="text-center text-slate-400 mt-20">
                            Chưa có cuộc hội thoại. Nhập mô tả tình huống để bắt đầu.
                        </div>
                    </template>

                    <template x-for="(msg, idx) in conversation" :key="idx">
                        <div class="mb-3 flex" :class="msg.from === 'user' ? 'justify-end' : 'justify-start'">
                            <div :class="msg.from === 'user' 
                                    ? 'bg-indigo-600 text-white' 
                                    : 'bg-white text-slate-800 shadow'"
                                 class="max-w-[80%] p-3 rounded-lg">
                                <div class="text-sm" x-text="msg.text"></div>
                                <div class="text-[10px] text-slate-400 mt-1 text-right" x-text="msg.time"></div>
                            </div>
                        </div>
                    </template>
                </div>

                <!-- Input + Gửi + Xóa -->
                <div class="mt-4 flex gap-3">
                    <input
                        x-model="inputText"
                        @keyup.enter="handleSend"
                        placeholder="Mô tả tình huống (ví dụ: Tôi đi xe máy vượt đèn đỏ ở ngã tư)"
                        class="flex-1 rounded-lg border p-3 bg-white"
                    />
                    <button @click="handleSend" :disabled="loading"
                            class="px-4 py-2 rounded-lg bg-indigo-600 text-white disabled:opacity-50">
                        <span x-show="!loading">Gửi</span>
                        <span x-show="loading">Đang xử lý...</span>
                    </button>
                    <button @click="clearAll" class="px-4 py-2 rounded-lg bg-slate-200">Xoá</button>
                </div>

                <!-- Câu hỏi từ hệ thống -->
                <template x-if="pendingQuestion">
                    <div class="mt-4 bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded">
                        <div class="font-medium mb-2">Hệ thống hỏi:</div>
                        <div class="mb-3" x-text="pendingQuestion.text"></div>
                        <div class="flex flex-wrap gap-2">
                            <template x-for="(option, i) in pendingQuestion.options" :key="i">
                                <button @click="handleAnswer(option)"
                                        class="px-3 py-1 rounded bg-white shadow-sm text-sm"
                                        x-text="option">
                                </button>
                            </template>
                        </div>
                    </div>
                </template>

                <!-- Kết quả cuối -->
                <template x-if="result">
                    <div class="mt-4 bg-green-50 border-l-4 border-green-400 p-3 rounded">
                        <div class="font-semibold">Kết quả cuối cùng</div>
                        <div class="mt-2" x-text="result.summary"></div>
                        <div class="text-sm text-slate-600 mt-2">Lý do:</div>
                        <ul class="text-sm list-disc list-inside">
                            <template x-for="reason in result.reason" :key="reason">
                                <li x-text="reason"></li>
                            </template>
                        </ul>
                        <div class="text-xs text-slate-500 mt-2" x-text="'Căn cứ pháp lý: ' + result.legal"></div>
                    </div>
                </template>
            </div>
        </div>

        <!-- Right: Facts + Quản lý -->
        <div class="col-span-5 space-y-4">
            <!-- Facts -->
            <div class="bg-white rounded-2xl shadow p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-semibold">Facts hiện tại</h3>
                    <button @click="facts = []; result = null" class="text-xs px-2 py-1 rounded bg-slate-100">
                        Xoá facts
                    </button>
                </div>

                <template x-if="facts.length === 0">
                    <div class="text-slate-400">Chưa có facts.</div>
                </template>

                <div class="space-y-2">
                    <template x-for="(f, i) in facts" :key="i">
                        <div class="flex items-center justify-between bg-gray-50 p-2 rounded">
                            <div class="text-sm">
                                <span class="font-medium" x-text="f.key"></span>: <span x-text="f.value"></span>
                            </div>
                            <button @click="facts.splice(i, 1)" class="text-xs text-red-500">X</button>
                        </div>
                    </template>
                </div>
            </div>

            <!-- Quản lý luật (preview) -->
            
        </div>
    </div>
</div>

<script>
function trafficAdvisor() {
    return {
        inputText: '',
        conversation: [],
        facts: [],
        pendingQuestion: null,
        result: null,
        loading: false,

        // Gửi tin nhắn người dùng
        handleSend() {
            if (!this.inputText.trim() || this.loading) return;

            const userMsg = {
                from: 'user',
                text: this.inputText,
                time: new Date().toLocaleTimeString('vi-VN')
            };
            this.conversation.push(userMsg);
            this.inputText = '';
            this.loading = true;

            // Mock xử lý sau 900ms
            setTimeout(() => {
                const text = userMsg.text.toLowerCase();

                if (text.includes('vượt đèn')) {
                    // Thêm facts cơ bản
                    this.facts.push({ key: 'action', value: 'vượt đèn đỏ' });
                    if (text.includes('xe máy') || text.includes('xem máy')) {
                        this.facts.push({ key: 'vehicle', value: 'xe máy' });
                    }

                    // Hỏi thêm
                    const question = {
                        id: Date.now(),
                        text: 'Tại vị trí bạn vượt đèn đỏ có biển báo gì không?',
                        options: ['Không có biển báo', 'Biển cấm rẽ trái', 'Đèn vàng nhấp nháy', 'Khác (nhập mô tả)']
                    };
                    this.pendingQuestion = question;
                    this.conversation.push({
                        from: 'system',
                        text: question.text,
                        time: new Date().toLocaleTimeString('vi-VN')
                    });
                } else {
                    this.facts.push({ key: 'note', value: userMsg.text });
                    const tentative = {
                        summary: 'Không xác định rõ hành vi - cần thêm thông tin',
                        details: []
                    };
                    this.result = tentative;
                    this.conversation.push({
                        from: 'system',
                        text: tentative.summary,
                        time: new Date().toLocaleTimeString('vi-VN')
                    });
                }

                this.loading = false;
            }, 900);
        },

        // Xử lý trả lời câu hỏi
        handleAnswer(option) {
            if (!this.pendingQuestion) return;

            this.conversation.push({
                from: 'user',
                text: option,
                time: new Date().toLocaleTimeString('vi-VN')
            });

            // Cập nhật facts
            if (option === 'Không có biển báo') {
                this.facts.push({ key: 'has_sign', value: 'false' });
            } else if (option === 'Biển cấm rẽ trái') {
                this.facts.push({ key: 'sign_type', value: 'no_turn_left' });
                this.facts.push({ key: 'has_sign', value: 'true' });
            } else if (option === 'Đèn vàng nhấp nháy') {
                this.facts.push({ key: 'signal', value: 'yellow_blink' });
                this.facts.push({ key: 'has_sign', value: 'true' });
            } else {
                this.facts.push({ key: 'sign_desc', value: option });
            }

            this.pendingQuestion = null;
            this.loading = true;

            setTimeout(() => {
                const final = {
                    summary: 'Kết luận: Vi phạm vượt đèn đỏ - áp dụng mức phạt 600.000 VNĐ',
                    reason: [
                        'Phương tiện: xe máy',
                        'Hành vi: vượt đèn đỏ',
                        'Biển báo: đã xác định'
                    ],
                    legal: 'Nghị định 100/2019/NĐ-CP'
                };
                this.result = final;
                this.conversation.push({
                    from: 'system',
                    text: final.summary,
                    time: new Date().toLocaleTimeString('vi-VN')
                });
                this.loading = false;
            }, 900);
        },

        // Xóa toàn bộ
        clearAll() {
            this.conversation = [];
            this.facts = [];
            this.pendingQuestion = null;
            this.result = null;
            this.inputText = '';
        }
    };
}
</script>

</body>
</html>