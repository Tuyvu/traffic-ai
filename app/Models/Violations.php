<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Violations extends Model
{
    use HasFactory;

    protected $table = 'violations';

    protected $fillable = [
        'keyword',
        'normalized_form',
        'related_rule_id',
        'synonyms',
    ];

    protected $casts = [
        'synonyms' => 'array',
    ];

    /**
     * Related rule (if a Rule model exists in App\Models\Rule)
     */
    public function relatedRule()
    {
        return $this->belongsTo(\App\Models\Rule::class, 'related_rule_id');
    }
}
