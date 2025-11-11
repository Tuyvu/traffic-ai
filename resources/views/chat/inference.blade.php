<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tư vấn Vi phạm Giao thông - AI Luật</title>
    <script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        [x-cloak] { display: none !important; }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">

<div class="container mx-auto p-4 max-w-7xl" x-data="trafficChat()" x-init="init()">
    
    <div class="bg-white rounded-3xl shadow-2xl overflow-hidden">
        <div class="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6">
            <h1 class="text-2xl font-bold text-center">Tư vấn Vi phạm Giao thông Thông minh</h1>
            <p class="text-center mt-2 opacity-90">Mô tả hành vi → Nhận kết quả phạt ngay lập tức!</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-0">
            <!-- Chat Area -->
            <div class="lg:col-span-2 flex flex-col h-[680px]">
                <div class="flex-1 overflow-y-auto p-6 space-y-4" id="chatContainer">
                    <template x-if="messages.length === 0">
                        <div class="text-center text-gray-500 mt-20">
                            <p class="text-lg">Xin chào! Hãy mô tả hành vi của bạn nhé</p>
                            <p class="text-sm mt-2">Ví dụ: "Tôi đi xe máy vượt đèn đỏ ở ngã tư có biển cấm"</p>
                        </div>
                    </template>

                    <template x-for="msg in messages" :key="msg.id">
                        <div class="flex" :class="msg.role === 'user' ? 'justify-end' : 'justify-start'">
                            <div :class="msg.role === 'user' 
                                ? 'bg-indigo-600 text-white' 
                                : 'bg-gray-100 text-gray-800'"
                                class="max-w-xs lg:max-w-md px-4 py-3 rounded-2xl shadow">
                                <p x-text="msg.content"></p>
                                <span class="text-xs opacity-70 block mt-1 text-right">
                                    x-text="msg.time"
                                </span>
                            </div>
                        </div>
                    </template>
                </div>

                <!-- Pending Question -->
                <div x-show="currentQuestion && currentQuestion.question" x-cloak class="bg-yellow-50 border-t-4 border-yellow-400 p-4">
                    <p class="font-semibold text-yellow-800">Hệ thống cần hỏi thêm:</p>
                    <p class="mt-2 text-lg" x-text="currentQuestion.question"></p>
                    <div class="mt-3 flex flex-wrap gap-3" x-show="currentQuestion.options.length > 0">
                        <template x-for="option in currentQuestion.options" :key="option">
                            <button @click="answerQuestion(option)"
                                class="px-4 py-2 bg-white border border-yellow-400 rounded-lg hover:bg-yellow-100 transition">
                                <span x-text="option"></span>
                            </button>
                        </template>
                    </div>
                </div>

                <!-- Final Result -->
                <div x-show="finalResult" x-cloak class="bg-green-50 border-t-4 border-green-500 p-6">
                    <h3 class="text-xl font-bold text-green-800">KẾT LUẬN PHẠT</h3>
                    <template x-for="result in finalResult">
                        <div class="mt-4 p-4 bg-white rounded-xl shadow">
                            <p class="text-lg font-semibold" x-text="result.title"></p>
                            <p class="mt-2 text-2xl font-bold text-red-600">
                                Phạt: <span x-text="result.penalty?.fine"></span>
                            </p>
                            <p class="text-sm text-gray-600 mt-2" x-text="result.conclusion"></p>
                            <p class="text-xs text-gray-500 mt-3">
                                Căn cứ: <span x-text="result.code"></span>
                            </p>
                        </div>
                    </template>
                </div>

                <!-- Input -->
                <div class="p-4 border-t bg-gray-50">
                    <div class="flex gap-3">
                        <input x-model="userInput" @keyup.enter="sendMessage"
                            placeholder="Nhập mô tả hành vi..."
                            class="flex-1 px-4 py-3 rounded-xl border focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            :disabled="loading"/>
                        <button @click="sendMessage" :disabled="loading || !userInput.trim()"
                            class="px-6 py-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 transition">
                            <span x-show="!loading">Gửi</span>
                            <span x-show="loading">Đang phân tích...</span>
                        </button>
                        <button @click="resetChat" class="px-4 py-3 bg-red-500 text-white rounded-xl hover:bg-red-600">
                            Reset
                        </button>
                    </div>
                </div>
            </div>

            <!-- Sidebar: Facts -->
            <div class="bg-gray-50 p-6 border-l">
                <h3 class="font-bold text-lg mb-4">Facts đã xác định</h3>
                <div class="space-y-3">
                    <template x-for="(fact, key) in facts" :key="key">
                        <div class="bg-white p-3 rounded-lg shadow text-sm">
                            <span class="font-medium text-indigo-600" x-text="key"></span>:
                            <span x-text="fact.join(', ')"></span>
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
function trafficChat() {
    return {
        userInput: '',
        messages: [],
        facts: {},
        currentQuestion: null,
        finalResult: null,
        loading: false,

        init() {
            this.scrollToBottom();
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const container = this.$refs.chatContainer || document.getElementById('chatContainer');
                if (container) container.scrollTop = container.scrollHeight;
            });
        },

        addMessage(content, role = 'assistant') {
            this.messages.push({
                id: Date.now(),
                content,
                role,
                time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })
            });
            this.scrollToBottom();
        },

        sendMessage() {
            if (!this.userInput.trim() || this.loading) return;
            const text = this.userInput.trim();
            this.addMessage(text, 'user');
            this.userInput = '';
            this.loading = true;
            console.log(text);

            fetch("{{ route('inference.infer') }}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({ text })
            })
            .then(r => r.json())
            .then(data => {
                console.log(data);
                this.handleResponse(data);
            })
            .catch(() => {
                this.addMessage('Lỗi kết nối hệ thống. Vui lòng thử lại!');
            })
            .finally(() => {
                this.loading = false;
            });
        },

        answerQuestion(option) {
            this.addMessage(option, 'user');
            this.loading = true;

            fetch("{{ route('inference.infer') }}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({
                    answer: option,
                    slot: this.currentQuestion.slot
                })
            })
            .then(r => r.json())
            .then(data => {
                this.handleResponse(data);
            })
            .finally(() => {
                this.loading = false;
            });
        },

        handleResponse(data) {
    if (data.error) {
        this.addMessage('Error: ' + data.error);
        return;
    }

    // CẬP NHẬT FACTS
    if (data.facts) {
        this.facts = { ...this.facts, ...data.facts };
    }

    // RESET currentQuestion VỀ OBJECT RỖNG (KHÔNG PHẢI null)
    this.currentQuestion = { question: '', options: [], slot: '' };

    if (data.status === 'need_info' && data.questions && data.questions.length > 0) {
        const q = data.questions[0];
        this.currentQuestion = {
            slot: q.slot || '',
            question: q.question || 'Bạn có thể nói rõ hơn không?',
            options: Array.isArray(q.options) ? q.options : []
        };
        this.addMessage(this.currentQuestion.question);
    }

    if (data.status === 'result' && data.results && data.results.length > 0) {
        this.finalResult = data.results;
        this.currentQuestion = { question: '', options: [], slot: '' }; // reset
        data.results.forEach(r => {
            const fine = r.penalty?.fine || 'Chưa xác định';
            const code = r.code || 'Nghị định 100/2019';
            this.addMessage(`
                <strong class="text-lg">${r.title}</strong><br>
                <span class="text-2xl font-bold text-red-600">Phạt: ${fine} VNĐ</span><br>
                <span class="text-sm text-gray-600">${r.conclusion || ''}</span><br>
                <span class="text-xs text-gray-500">Căn cứ: ${code}</span>
            `);
        });
    }

    if (data.status === 'unknown') {
        this.addMessage(data.questions?.[0]?.question || "Mình chưa hiểu rõ. Bạn mô tả lại nhé!");
    }
},

        resetChat() {
            if (!confirm('Xóa toàn bộ cuộc trò chuyện?')) return;
            this.messages = [];
            this.facts = {};
            this.currentQuestion = null;
            this.finalResult = null;
            fetch("{{ route('inference.reset') }}", {
                method: 'POST',
                headers: { 'X-CSRF-TOKEN': '{{ csrf_token() }}' }
            });
        }
    }
}
</script>

</body>
</html>