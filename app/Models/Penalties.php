<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Penalties extends Model
{
    use HasFactory;

    protected $table = 'penalties';

    protected $fillable = [
        'law_ref',
        'article',
        'description',
        'fine_min',
        'fine_max',
        'unit',
        'additional_punishment',
    ];

    protected $casts = [
        'fine_min' => 'decimal:2',
        'fine_max' => 'decimal:2',
    ];
}

