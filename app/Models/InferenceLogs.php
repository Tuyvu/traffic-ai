<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class InferenceLogs extends Model
{
    use HasFactory;

    protected $table = 'inference_logs';

    protected $fillable = [
        'session_id',
        'user_input',
        'facts',
        'missing_conditions',
        'result',
        'created_at',
    ];

    protected $casts = [
        'facts' => 'array',
        'missing_conditions' => 'array',
        'result' => 'array',
        'created_at' => 'datetime',
    ];
}
