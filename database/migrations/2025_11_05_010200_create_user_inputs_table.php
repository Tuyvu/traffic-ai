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
        Schema::create('user_inputs', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('session_id', 64)->index()->comment('Liên kết với log suy luận');
            $table->enum('from', ['user', 'system'])->default('user')->comment('Ai gửi tin');
            $table->text('message')->nullable()->comment('Nội dung câu hỏi/trả lời');
            $table->dateTime('timestamp')->useCurrent()->comment('Thời gian gửi');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('user_inputs');
    }
};
