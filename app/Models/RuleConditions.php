<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class RuleConditions extends Model
{
    use HasFactory;

    protected $fillable = [
        'rule_id',
        'condition_type',
        'condition_value',
    ];
}
