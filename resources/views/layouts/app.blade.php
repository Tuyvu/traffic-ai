<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://unpkg.com/mammoth/mammoth.browser.min.js"></script>

    <title>Hệ thống suy diễn tri thức</title>

    <link rel="stylesheet" href="{{ asset('css/app.css') }}">
</head>
<body>
    
    <header class="main-header">
        <h1>HỆ THỐNG SUY DIỄN TRI THỨC</h1>
    </header>

    <main class="main-container">
        @yield('content')
    </main>

    <script src="{{ asset('js/app.js') }}" defer></script>

</body>
</html>