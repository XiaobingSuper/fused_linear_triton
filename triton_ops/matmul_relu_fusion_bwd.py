import torch

from triton import Config, autotune, cdiv, heuristics, jit
from triton import language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


from .activation import relu_grad

def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                               num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(
                            Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                   num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@jit
def _kernel(A, B, C, GRAD_ACT, SAVED_Y, M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            stridegrad_act_m, stridegrad_act_k,  #
            stride_saved_act_y_m, stride_saved_act_y_k,  #
            dot_out_dtype: tl.constexpr,  #
            allow_tf32: tl.constexpr,  #
            fp8_fast_accum: tl.constexpr,  #
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, AB_DTYPE: tl.constexpr  #
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    
    GRAD_ACT = GRAD_ACT + (ram[:, None] * stridegrad_act_m + rk[None, :] * stridegrad_act_k)
    SAVED_Y = SAVED_Y + (ram[:, None] * stride_saved_act_y_m + rk[None, :] * stride_saved_act_y_k)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
            saved_y = tl.load(SAVED_Y)
            # grad_act = a * relu_grad(saved_y)
            grad_act = tl.where(saved_y > 0, a, tl.zeros_like(a))
            tl.store(GRAD_ACT, grad_act)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
            saved_y = tl.load(SAVED_Y, mask=rk[None, :] < k_remaining, other=_0)
            # grad_act = a * relu_grad(saved_y)
            grad_act = tl.where(saved_y > 0, a, tl.zeros_like(a))
            tl.store(GRAD_ACT, grad_act, mask=rk[None, :] < k_remaining)
        
        if AB_DTYPE:
            # a = a.to(C.dtype.element_ty)
            grad_act = grad_act.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if fp8_fast_accum:
            acc = tl.dot(grad_act, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(grad_act, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
        GRAD_ACT += BLOCK_K * SPLIT_K * stridegrad_act_k
        SAVED_Y += BLOCK_K * SPLIT_K * stride_saved_act_y_k

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


def matmul_relu_fusion_backward(a, b, saved_act_y, dot_out_dtype=None, allow_tf32=True, fp8_fast_accum=True):
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    if a.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5] or\
        b.dtype in [tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
        c_dtype = torch.float16
    elif a.dtype in [torch.int8] or b.dtype in [torch.int8]:
        c_dtype = torch.int32
    else:
        c_dtype = get_higher_dtype(a.dtype, b.dtype)

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    grad_act = torch.empty_like(a)
    assert saved_act_y.dtype == a.dtype, "saved_act_y must have the same dtype as a"

    if dot_out_dtype is None:
        if c_dtype in [torch.float16, torch.float32, torch.bfloat16]:
            dot_out_dtype = tl.float32
        else:
            dot_out_dtype = tl.int32
    else:
        assert isinstance(dot_out_dtype, torch.dtype), "dot_out_dtype must be a torch.dtype"
        if dot_out_dtype == torch.float16:
            dot_out_dtype = tl.float16
        elif dot_out_dtype in [torch.float32, torch.bfloat16]:
            dot_out_dtype = tl.float32
        else:
            dot_out_dtype = tl.int32

    ab_dtype = True
    if a.dtype in [tl.float8e4nv, tl.float8e5] and b.dtype in [tl.float8e4nv, tl.float8e5]:
        ab_dtype = False
    if a.dtype in [torch.int8] and b.dtype in [torch.int8]:
        ab_dtype = False
    # launch kernel
    grid = lambda META: (cdiv(M, META['BLOCK_M']) * cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    _kernel[grid](
        a, b, c, grad_act, saved_act_y, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        grad_act.stride(0), grad_act.stride(1),  #
        saved_act_y.stride(0), saved_act_y.stride(1),  #
        dot_out_dtype=dot_out_dtype,  #
        allow_tf32=allow_tf32,  #
        fp8_fast_accum=fp8_fast_accum,  #
        GROUP_M=8, AB_DTYPE=ab_dtype)
    return c, grad_act
    