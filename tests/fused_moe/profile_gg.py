import torch
import random
import torch.profiler
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm

def random_partition(size_a: int, target: int, randomp: bool):
    if randomp:
        cuts = sorted(random.sample(range(target + size_a - 1), size_a - 1))
        cuts = [-1] + cuts + [target + size_a - 1]
        result = [cuts[i + 1] - cuts[i] - 1 for i in range(size_a)]
        return result
    else:
        return [target//size_a] * size_a

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def calculate_gg_mem(input, expert, output):
    bytes_read_input = input.numel() * input.element_size()
    bytes_read_expert = expert.numel() * expert.element_size()
    bytes_write_output = output.numel() * output.element_size()
    total_bytes = (
        bytes_read_input +
        bytes_read_expert +
        bytes_write_output
    )
    # Bytes -> GB
    return total_bytes / (1024 ** 3)


def test_moe_gemm(n_experts, intermediate_size, hidden_size, tokens, topk, dtype, tp, gemm1=True):

    iterations = 10

    total_m = tokens * topk

    input_a1 = torch.randn(total_m, hidden_size, dtype=dtype, device="xpu") / 10
    w13 = torch.randn(n_experts, 2*intermediate_size//tp, hidden_size, dtype=dtype, device="xpu") / 10
    w2 = torch.randn(n_experts, hidden_size, intermediate_size//tp, dtype=dtype, device="xpu") / 10
    w13 = w13.transpose(1, 2).contiguous().transpose(1, 2)
    w2 = w2.transpose(1, 2).contiguous().transpose(1, 2)

    token_per_group = random_partition(n_experts, total_m, True)
    bias = None
    print("distribution:" , token_per_group)

    with torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU  # ✅ 使用 XPU
        ],
        # record_shapes=True,
        # with_stack=False,
        # profile_memory=True,
        with_flops=True,
    ) as prof:

        cutlass_o1 = torch.empty((total_m, 2*intermediate_size//tp), dtype=dtype, device="xpu")
        cutlass_o3 = torch.empty((total_m, hidden_size), dtype=dtype, device="xpu")
        if gemm1:
            input = input_a1
            weights = w13
            output = cutlass_o1
            N = 2*intermediate_size // tp
            K = hidden_size
        else:
            cutlass_o2 = (cutlass_o1[:, :(intermediate_size//tp)])
            input = cutlass_o2
            weights = w2
            output = cutlass_o3
            N = hidden_size
            K = intermediate_size // tp

        for _ in range(iterations):
            cutlass_grouped_gemm(input, weights, bias, output, token_per_group, N, K, n_experts)

    print(prof.key_averages().table(
        sort_by="self_xpu_time_total",  # 以XPU耗时排序
        row_limit=-1
    ))
    print("total mem(GB):", calculate_gg_mem(input, weights[:count_nonzero(token_per_group)], output))

if __name__ == "__main__":
    # llama-4-scout
    # prefill
    # test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=8192, gemm1=True)
    # decode
    # test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=8, gemm1=True)
    # test_moe_gemm(n_experts=16, intermediate_size=8192, hidden_size=5120, topk=1, dtype=torch.bfloat16, tokens=8, gemm1=True)
    model = "qwen-3-30b-a3b"
    if model == "qwen-3-30b-a3b":
        tokens      = 80
        hidden_size = 2048
        inter_size  = 768
        experts     = 128
        topk        = 8
        tp          = 4

    test_moe_gemm(n_experts=experts, intermediate_size=inter_size, hidden_size=hidden_size, topk=topk, dtype=torch.bfloat16, tokens=tokens, tp=tp, gemm1=False)

