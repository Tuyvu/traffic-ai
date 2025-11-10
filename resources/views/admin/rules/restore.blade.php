@extends('layouts.app')

@section('content')
<div class="d-flex justify-content-between align-items-center mb-3">
    <h3>Danh sách luật</h3>
    <a href="{{ route('rules.create') }}" class="btn btn-primary">+ Thêm luật</a>
</div>

<table class="table table-bordered table-hover align-middle">
    <thead class="table-light">
        <tr>
            <th width="50">#</th>
            <th>Mã luật</th>
            <th>Mô tả</th>
            <th>Phân loại</th>
            <th>Phạt min-max</th>
            <th width="130">Hành động</th>
        </tr>
    </thead>

    <tbody>
        @forelse($rules as $rule)
        <tr>
            <td>{{ $loop->iteration + ($rules->currentPage()-1)*$rules->perPage() }}</td>
            <td><strong>{{ $rule->code }}</strong></td>
            <td>{{ $rule->description }}</td>
            <td>{{ $rule->category }}</td>
            <td>{{ number_format($rule->penalty_min) }} - {{ number_format($rule->penalty_max) }}</td>
            <td>
                <a href="{{ route('rules.edit', $rule) }}" class="btn btn-warning btn-sm">Sửa</a>

                <form action="{{ route('rules.destroy', $rule) }}" method="POST" style="display:inline-block" 
                    onsubmit="return confirm('Xóa luật này?');">
                    @csrf
                    @method('DELETE')
                    <button class="btn btn-danger btn-sm">Xóa</button>
                </form>
            </td>
        </tr>
        @empty
        <tr>
            <td colspan="6" class="text-center">Chưa có luật nào.</td>
        </tr>
        @endforelse
    </tbody>
</table>

<div class="mt-3">
    {{ $rules->links() }}
</div>
@endsection
