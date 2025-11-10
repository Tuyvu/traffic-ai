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
        Schema::create('violations', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('keyword', 255)->comment('Từ khóa NLP nhận dạng được');
            $table->string('normalized_form', 255)->nullable()->comment('Dạng chuẩn');
            $table->unsignedBigInteger('related_rule_id')->nullable()->comment('Luật liên quan');
            $table->json('synonyms')->nullable()->comment('Danh sách từ đồng nghĩa');
            $table->timestamps();

            // foreign key to rules table if exists
            if (Schema::hasTable('rules')) {
                $table->foreign('related_rule_id')->references('id')->on('rules')->onDelete('set null');
            }
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('violations');
    }
};
