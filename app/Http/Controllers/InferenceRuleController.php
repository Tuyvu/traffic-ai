<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Rules;
use App\Models\SetRule;
use Illuminate\Support\Facades\Http;
// use App\Helpers\Formula;

class InferenceRuleController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        return view('chat.inference');
    }
    public function infer(Request $req) 
    {
        // dd($req->all());
        $sessionId = $req->session()->getId();
        $payload = [
            'session_id' => $sessionId,
            'text' => $req->input('text'), // or 'facts'
        ];
        // $resp = Http::post(env('CORE_URL').'/infer', $payload);
        $resp = Http::withHeaders([
            'Content-Type' => 'application/json',
            'Accept' => 'application/json',
            ])->post('http://service-appsra:5000/infer', $payload);
        return response()->json($resp->json());
    }

}
