<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tư vấn Vi phạm Giao thông AI</title>
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">

<div class="max-w-6xl mx-auto p-6" x-data="trafficBot()" x-init="init()">
    <div class="grid grid-cols-12 gap-6">

        <!-- CHAT TRÁI -->
        <div class="col-span-8">
            <div class="bg-white rounded-2xl shadow-xl overflow-hidden">
                <div class="bg-gradient-to-r from-indigo-600 to-purple-600 p-5 text-white">
                    <h1 class="text-2xl font-bold text-center">Tư vấn Vi phạm Giao thông AI</h1>
                </div>

                <div class="h-96 overflow-y-auto p-5 space-y-4" id="chatBox">
                    <template x-for="msg in messages" :key="msg.id">
                        <div class="flex" :class="msg.role === 'user' ? 'justify-end' : 'justify-start'">
                            <div :class="msg.role === 'user' 
                                ? 'bg-indigo-600 text-white' 
                                : 'bg-gray-100 text-gray-800'"
                                class="max-w-lg px-5 py-3 rounded-2xl shadow-md">
                                <div x-html="msg.content"></div>
                                <div class="text-xs opacity-70 mt-1" x-text="msg.time"></div>
                            </div>
                        </div>
                    </template>
                </div>

                <!-- CÂU HỎI + NÚT BẤM -->
                <div x-show="question" class="bg-yellow-50 border-t-4 border-yellow-400 p-5">
                    <p class="font-semibold text-yellow-900 mb-3" x-text="question.question"></p>
                    <div class="flex flex-wrap gap-3">
                        <template x-for="opt in question.options">
                            <button @click="selectOption(opt)" 
                                class="px-5 py-3 bg-white border-2 border-yellow-400 rounded-xl hover:bg-yellow-100 transition font-medium"
                                x-text="opt">
                            </button>
                        </template>
                    </div>
                </div>

                <!-- KẾT LUẬN -->
                <div x-show="result" class="bg-green-50 border-t-4 border-green-500 p-6">
                    <h3 class="text-2xl font-bold text-green-800 mb-4">KẾT LUẬN PHẠT</h3>
                    <template x-for="v in result.violations">
                        <div class="bg-white rounded-xl shadow-lg p-5 mb-4">
                            <h4 class="text-xl font-bold text-red-600" x-text="v.title"></h4>
                            <p class="text-3xl font-black text-red-700 mt-2" x-text="v.penalty"></p>
                            <p class="text-sm text-gray-600 mt-2" x-text="v.description"></p>
                            <p class="text-xs font-medium text-gray-500 mt-3">
                                Căn cứ: <span x-text="v.legal_ref"></span>
                            </p>
                        </div>
                    </template>
                    <p class="text-center font-bold text-green-700">
                        Tổng phạt: <span x-text="result.total || 'Đang tính...'"></span>
                    </p>
                </div>

                <!-- INPUT -->
                <div class="p-5 border-t bg-gray-50">
                    <div class="flex gap-3">
                        <input x-model="input" @keyup.enter="send()" 
                            placeholder="Mô tả hành vi của bạn..." 
                            class="flex-1 px-5 py-4 rounded-xl border focus:ring-4 focus:ring-indigo-300 outline-none"
                            :disabled="loading">
                        <button @click="send()" :disabled="loading || !input.trim()"
                            class="px-8 py-4 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 font-bold">
                            <span x-show="!loading">Gửi</span>
                            <span x-show="loading">Đang xử lý...</span>
                        </button>
                        <button @click="reset()" class="px-6 py-4 bg-red-500 text-white rounded-xl hover:bg-red-600">
                            Reset
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- FACTS PHẢI -->
        <div class="col-span-4">
            <div class="bg-white rounded-2xl shadow-xl p-6">
                <h3 class="text-xl font-bold mb-4 text-indigo-600">Facts đã xác định</h3>
                <div class="space-y-3">
                    <template x-for="(values, key) in facts">
                        <div class="bg-indigo-50 p-4 rounded-lg">
                            <span class="font-bold text-indigo-700" x-text="key"></span>:
                            <span class="font-medium" x-text="values.join(', ')"></span>
                        </div>
                    </template>
                    <div x-show="Object.keys(facts).length === 0" class="text-gray-400 text-center">
                        Chưa có thông tin
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
function trafficBot() {
    return {
        input: '',
        messages: [],
        facts: {},
        question: null,
        result: null,
        loading: false,
        sessionId: '{{ session()->getId() }}',

        init() {
            this.addMsg('Xin chào! Hãy mô tả hành vi bạn vừa làm nhé<br><small class="opacity-70">Ví dụ: Tôi đi xe máy vượt đèn đỏ có biển cấm</small>', 'assistant');
        },

        addMsg(content, role = 'assistant') {
            this.messages.push({
                id: Date.now(),
                content,
                role,
                time: new Date().toLocaleTimeString('vi', {hour: '2-digit', minute: '2-digit'})
            });
            this.$nextTick(() => {
                document.getElementById('chatBox').scrollTop = document.getElementById('chatBox').scrollHeight;
            });
        },

        send() {
            if (!this.input.trim() || this.loading) return;
            const text = this.input.trim();
            this.addMsg(text, 'user');
            this.input = '';
            this.loading = true;

            fetch("{{ route('inference.infer') }}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({ 
                    session_id: this.sessionId,
                    message: text 
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.session_facts) this.facts = data.session_facts;
                if (data.status === 'need_info' && data.questions?.length > 0) {
                    this.question = data.questions[0];
                    this.addMsg(this.question.question);
                }
                if (data.status === 'result') {
                    this.result = data;
                    this.question = null;
                    const total = data.violations.reduce((s,v) => {
                        const match = v.penalty.match(/([\d,]+)-([\d,]+)/);
                        if (match) {
                            const min = parseFloat(match[1].replace(/,/g, ''));
                            const max = parseFloat(match[2].replace(/,/g, ''));
                            const avg = (min + max) / 2;
                            return s + avg;
                        }
                        return s + parseFloat(v.penalty.replace(/,/g, '')) || 0;
                    }, 0);
                    data.total = `${(total).toLocaleString('vi')} đồng`;
                }
                if (data.status === 'unknown') {
                    this.addMsg(data.message || 'Mình chưa hiểu, bạn nói rõ hơn nhé!');
                }
            })
            .finally(() => this.loading = false);
        },

        selectOption(opt) {
            this.addMsg(opt, 'user');
            this.question = null;
            this.loading = true;

            fetch("{{ route('inference.infer') }}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({ 
                    session_id: this.sessionId,
                    message: opt 
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.session_facts) this.facts = data.session_facts;
                if (data.status === 'result') {
                    this.result = data;
                    const total = data.violations.reduce((s,v) => {
                        const match = v.penalty.match(/([\d,]+)-([\d,]+)/);
                        if (match) {
                            const min = parseFloat(match[1].replace(/,/g, ''));
                            const max = parseFloat(match[2].replace(/,/g, ''));
                            const avg = (min + max) / 2;
                            return s + avg;
                        }
                        return s + parseFloat(v.penalty.replace(/,/g, '')) || 0;
                    }, 0);
                    data.total = `${total.toLocaleString('vi')} đồng`;
                }
            })
            .finally(() => this.loading = false);
        },

        reset() {
            if (!confirm('Xóa toàn bộ cuộc trò chuyện?')) return;
            fetch("{{ route('inference.reset') }}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                }
            })
            this.messages = [];
            this.facts = {};
            this.question = null;
            this.result = null;
            this.addMsg('Đã reset! Hãy bắt đầu lại nhé', 'assistant');
        }
    }
}
</script>
</body>
</html>