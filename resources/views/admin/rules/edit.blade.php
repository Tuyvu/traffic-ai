@extends('admin.layouts')

@section('content')
<h3 class="mb-4">{{ isset($rule) ? 'Sửa luật' : 'Thêm luật mới' }}</h3>

<form method="POST" action="{{ isset($rule) ? route('rules.update',$rule) : route('rules.store') }}">
    @csrf
    @if(isset($rule)) @method('PUT') @endif

    <ul class="nav nav-tabs mb-3" id="ruleTabs">
        <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#info">Thông tin luật</a></li>
        <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#cond">Điều kiện</a></li>
        <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#res">Kết luận</a></li>
    </ul>

    <div class="tab-content">

        {{-- TAB 1 --}}
        <div class="tab-pane fade show active" id="info">
            <div class="mb-3">
                <label>Mã luật</label>
                <input name="code" class="form-control" value="{{ $rule->code ?? '' }}">
            </div>

            <div class="mb-3">
                <label>Mô tả</label>
                <textarea name="description" class="form-control">{{ $rule->description ?? '' }}</textarea>
            </div>

            <div class="row">
                <div class="col">
                    <label>Phạt tối thiểu</label>
                    <input name="penalty_min" type="number" class="form-control" value="{{ $rule->penalty_min ?? '' }}">
                </div>
                <div class="col">
                    <label>Phạt tối đa</label>
                    <input name="penalty_max" type="number" class="form-control" value="{{ $rule->penalty_max ?? '' }}">
                </div>
            </div>

            <div class="mt-3">
                <label>Trích dẫn luật</label>
                <input name="legal_ref" class="form-control" value="{{ $rule->legal_ref ?? '' }}">
            </div>

            <div class="mt-3">
                <label>Phân loại</label>
                <input name="category" class="form-control" value="{{ $rule->category ?? '' }}">
            </div>
        </div>

        {{-- TAB 2 --}}
        <div class="tab-pane fade" id="cond">
            <table class="table" id="condTable">
                <thead>
                    <tr><th>Field</th><th>Operator</th><th>Value</th><th></th></tr>
                </thead>
                <tbody>
                    @foreach($rule->conditions ?? [ ['field'=>'','operator'=>'','value'=>''] ] as $i => $c)
                    <tr>
                        <td><input name="conditions[{{ $i }}][field]" class="form-control" value="{{ $c->field ?? '' }}"></td>
                        <td><input name="conditions[{{ $i }}][operator]" class="form-control" value="{{ $c->operator ?? '' }}"></td>
                        <td><input name="conditions[{{ $i }}][value]" class="form-control" value="{{ $c->value ?? '' }}"></td>
                        <td><button type="button" class="btn btn-danger btn-sm remove-cond">X</button></td>
                    </tr>
                    @endforeach
                </tbody>
            </table>

            <button type="button" class="btn btn-primary btn-sm" id="addCond">+ Thêm điều kiện</button>
        </div>

        {{-- TAB 3 --}}
        <div class="tab-pane fade" id="res">
            <table class="table" id="resTable">
                <thead>
                    <tr><th>Key</th><th>Value</th><th></th></tr>
                </thead>
                <tbody>
                    @foreach($rule->results ?? [ ['result_key'=>'','result_value'=>''] ] as $i => $r)
                    <tr>
                        <td><input name="results[{{ $i }}][result_key]" class="form-control" value="{{ $r->result_key ?? '' }}"></td>
                        <td><input name="results[{{ $i }}][result_value]" class="form-control" value="{{ $r->result_value ?? '' }}"></td>
                        <td><button type="button" class="btn btn-danger btn-sm remove-res">X</button></td>
                    </tr>
                    @endforeach
                </tbody>
            </table>

            <button type="button" class="btn btn-primary btn-sm" id="addRes">+ Thêm kết luận</button>
        </div>

    </div>

    <button class="btn btn-success mt-3">Lưu</button>
</form>
<script>
document.getElementById('addCond').onclick = () => {
    let i = document.querySelectorAll('#condTable tbody tr').length;
    document.querySelector('#condTable tbody').insertAdjacentHTML('beforeend',
        `<tr>
            <td><input name="conditions[${i}][field]" class="form-control"></td>
            <td><input name="conditions[${i}][operator]" class="form-control"></td>
            <td><input name="conditions[${i}][value]" class="form-control"></td>
            <td><button type="button" class="btn btn-danger btn-sm remove-cond">X</button></td>
        </tr>`
    );
};

document.getElementById('addRes').onclick = () => {
    let i = document.querySelectorAll('#resTable tbody tr').length;
    document.querySelector('#resTable tbody').insertAdjacentHTML('beforeend',
        `<tr>
            <td><input name="results[${i}][result_key]" class="form-control"></td>
            <td><input name="results[${i}][result_value]" class="form-control"></td>
            <td><button type="button" class="btn btn-danger btn-sm remove-res">X</button></td>
        </tr>`
    );
};

document.addEventListener('click', e => {
    if (e.target.classList.contains('remove-cond') || e.target.classList.contains('remove-res')) {
        e.target.closest('tr').remove();
    }
});
</script>
