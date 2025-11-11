<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\InferenceRuleController;
use App\Http\Controllers\Admin\RuleController;


Route::get('/chat', [InferenceRuleController::class, 'index'])->name('inference.show');
Route::post('/chat/infer', [InferenceRuleController::class, 'infer'])->name('inference.infer');
Route::post('/chat/reset', [InferenceRuleController::class, 'reset'])->name('inference.reset');
Route::resource('rules', RuleController::class);
// Route để xử lý logic khi form được gửi
// [cite: 19, 50, 56]
// Route::post('/run-inference', [InferenceRuleController::class, 'store'])->name('inference.store');
// Route::post('/infer', [InferenceRuleController::class, 'store'])->name('infer.run');