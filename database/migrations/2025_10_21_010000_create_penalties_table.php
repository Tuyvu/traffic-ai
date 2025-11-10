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
        Schema::create('penalties', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('law_ref', 100)->nullable()->comment('Số hiệu nghị định');
            $table->string('article', 50)->nullable()->comment('Điều khoản');
            $table->text('description')->nullable()->comment('Mô tả hành vi');
            $table->decimal('fine_min', 10, 2)->nullable()->comment('Mức phạt tối thiểu');
            $table->decimal('fine_max', 10, 2)->nullable()->comment('Mức phạt tối đa');
            $table->string('unit', 10)->default('VNĐ')->comment('Đơn vị');
            $table->text('additional_punishment')->nullable()->comment('Hình phạt bổ sung');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('penalties');
    }
};
