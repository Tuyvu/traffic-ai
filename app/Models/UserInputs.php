<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class UserInputs extends Model
{
    use HasFactory;

    protected $table = 'user_inputs';

    protected $fillable = [
        'session_id',
        'from',
        'message',
        'timestamp',
    ];

    protected $casts = [
        'timestamp' => 'datetime',
    ];
}

