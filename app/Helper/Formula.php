<?php
private function convertFormula($formula)
{
    $converted = $formula;

    // 1️⃣ Đổi ký hiệu căn bậc hai
    $converted = str_replace(['√', '√('], ['sqrt(', 'sqrt('], $converted);

    // 2️⃣ Đổi các ký tự mũ (², ³, ⁴, …) sang **2, **3, …
    $converted = str_replace(
        ['²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'],
        ['**2', '**3', '**4', '**5', '**6', '**7', '**8', '**9'],
        $converted
    );

    // 3️⃣ Thay ký hiệu nhân ẩn: ab => a*b (trừ khi giữa chữ và hàm)
    // => thêm dấu * giữa chữ và chữ/số
    $converted = preg_replace('/(?<=[a-zA-Z])(?=[a-zA-Z0-9])/', '*', $converted);

    // 4️⃣ Đổi dấu nhân đặc biệt
    $converted = str_replace(['·', '×'], '*', $converted);

    // 5️⃣ Đổi các hàm lượng giác dùng độ sang radian
    $converted = preg_replace_callback('/(sin|cos|tan)\s*\(?([A-Za-z0-9]+)\)?/i', function ($matches) {
        $fn = strtolower($matches[1]);
        $arg = $matches[2];
        return "{$fn}(deg2rad($arg))";
    }, $converted);

    // 6️⃣ Dọn sạch dấu khoảng trắng thừa
    $converted = preg_replace('/\s+/', ' ', $converted);

    return trim($converted);
}
