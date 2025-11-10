<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Rules extends Model
{
    use HasFactory;

    protected $fillable = [
        'code',
        'title',
        'conclusion',
        'penalty_id',
        'source',
        'active',
    ];
}
