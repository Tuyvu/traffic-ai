<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('inference_logs', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('session_id', 64)->index()->comment('ID phiên tư vấn');
            $table->text('user_input')->nullable()->comment('Câu người dùng nhập');
            $table->json('facts')->nullable()->comment('Facts trích xuất từ NLP');
            $table->json('missing_conditions')->nullable()->comment('Điều kiện còn thiếu');
            $table->json('result')->nullable()->comment('Kết quả suy luận');
            $table->timestamp('created_at')->useCurrent()->comment('Thời gian ghi nhận');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('inference_logs');
    }
};
