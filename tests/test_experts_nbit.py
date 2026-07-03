import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Experts4bit, ExpertsNbit
from tests.helpers import get_available_devices
from tests.test_experts4bit import (
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    NUM_EXPERTS,
    _load_experts_lora,
    _random_expert_weights,
    _random_routing,
    _reference_forward,
)

# Per-scheme reconstruction tolerances: 8-bit blockwise is far tighter than 4-bit
# (rtol=0.15 in the 4-bit tests); 16-bit passthrough is a plain dtype cast.
RECON_TOLERANCES = {
    "int8": dict(rtol=0.03, atol=0.01),
    "fp8": dict(rtol=0.12, atol=0.03),
    "bf16": dict(rtol=0.02, atol=0.002),
    "fp16": dict(rtol=1e-3, atol=1e-4),
}

NEW_QUANT_TYPES = list(RECON_TOLERANCES)


def _storage_expectations(quant_type):
    """(packed dtype, packed elements per weight element) for each scheme."""
    if quant_type in ("nf4", "fp4"):
        return torch.uint8, 0.5
    if quant_type in ("int8", "fp8"):
        return torch.uint8, 1
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[quant_type], 1


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", NEW_QUANT_TYPES)
def test_expertsnbit_roundtrip(device, quant_type):
    torch.manual_seed(0)
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = ExpertsNbit.from_float(gate_up, down, quant_type=quant_type)

    storage_dtype, elems_per_value = _storage_expectations(quant_type)
    gate_up_out = 2 * INTERMEDIATE_DIM
    assert module.gate_up_proj.dtype == storage_dtype
    assert module.gate_up_proj.shape == (NUM_EXPERTS, int(gate_up_out * HIDDEN_DIM * elems_per_value))
    assert module.down_proj.shape == (NUM_EXPERTS, int(HIDDEN_DIM * INTERMEDIATE_DIM * elems_per_value))
    assert not module.gate_up_proj.requires_grad

    if module.bits < 16:
        assert module.gate_up_absmax.shape == (NUM_EXPERTS, gate_up_out * HIDDEN_DIM // module.blocksize)
    else:
        assert module.gate_up_absmax is None and module.down_absmax is None and module.code is None

    for e in range(NUM_EXPERTS):
        deq = module._dequantize_expert(
            module.gate_up_proj, module.gate_up_absmax, module._gate_up_shape, e, torch.float32
        )
        assert deq.shape == (gate_up_out, HIDDEN_DIM)
        assert deq.dtype == torch.float32
        torch.testing.assert_close(deq, gate_up[e], **RECON_TOLERANCES[quant_type])


@pytest.mark.parametrize("device", get_available_devices())
def test_expertsnbit_fidelity_ordering(device):
    # The point of the 8-bit scheme: reconstruction error well below 4-bit. Lock the
    # ordering in so a codebook/blocksize regression is loud.
    torch.manual_seed(0)
    gate_up, down = _random_expert_weights(torch.float32, device)

    def mean_abs_err(quant_type):
        module = ExpertsNbit.from_float(gate_up, down, quant_type=quant_type)
        err = 0.0
        for e in range(NUM_EXPERTS):
            deq = module._dequantize_expert(
                module.gate_up_proj, module.gate_up_absmax, module._gate_up_shape, e, torch.float32
            )
            err += (deq - gate_up[e]).abs().mean().item()
        return err / NUM_EXPERTS

    assert mean_abs_err("int8") < mean_abs_err("nf4") / 4


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", NEW_QUANT_TYPES)
def test_expertsnbit_forward_matches_reference(device, quant_type):
    # float32 compute so the only difference vs. the reference is float accumulation order.
    torch.manual_seed(0)
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = ExpertsNbit.from_float(gate_up, down, quant_type=quant_type, compute_dtype=torch.float32)

    hidden_states, top_k_index, top_k_weights = _random_routing(device)

    # Reference uses the exact weights the module holds internally (dequantized storage),
    # isolating forward/routing correctness from quantization error.
    gate_up_deq = torch.stack(
        [
            module._dequantize_expert(
                module.gate_up_proj, module.gate_up_absmax, module._gate_up_shape, e, torch.float32
            )
            for e in range(NUM_EXPERTS)
        ]
    )
    down_deq = torch.stack(
        [
            module._dequantize_expert(module.down_proj, module.down_absmax, module._down_shape, e, torch.float32)
            for e in range(NUM_EXPERTS)
        ]
    )

    ref = _reference_forward(gate_up_deq, down_deq, hidden_states, top_k_index, top_k_weights)
    out = module(hidden_states, top_k_index, top_k_weights)
    assert out.shape == hidden_states.shape
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", NEW_QUANT_TYPES)
def test_expertsnbit_state_dict_roundtrip(device, quant_type):
    torch.manual_seed(0)
    gate_up, down = _random_expert_weights(torch.float16, device)
    module = ExpertsNbit.from_float(gate_up, down, quant_type=quant_type, compute_dtype=torch.float16)

    sd = module.state_dict()
    assert "gate_up_proj" in sd and "down_proj" in sd
    assert "code" not in sd  # codebook is non-persistent (reconstructed at init)
    if module.bits < 16:
        assert "gate_up_absmax" in sd and "down_absmax" in sd
    else:
        assert "gate_up_absmax" not in sd and "down_absmax" not in sd

    reloaded = ExpertsNbit(
        NUM_EXPERTS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        quant_type=quant_type,
        compute_dtype=torch.float16,
        device=device,
    )
    result = reloaded.load_state_dict(sd, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []

    # Bit-exact restore of packed storage (+ absmax when present).
    torch.testing.assert_close(reloaded.gate_up_proj, module.gate_up_proj, rtol=0, atol=0)
    if module.bits < 16:
        torch.testing.assert_close(reloaded.down_absmax, module.down_absmax, rtol=0, atol=0)

    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    out_a = module(hidden_states, top_k_index, top_k_weights)
    out_b = reloaded(hidden_states, top_k_index, top_k_weights)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["int8", "bf16"])
def test_expertsnbit_backward_flows_to_input_and_base_stays_frozen(device, quant_type):
    torch.manual_seed(0)
    gate_up, down = _random_expert_weights(torch.float32, device)
    module = ExpertsNbit.from_float(gate_up, down, quant_type=quant_type, compute_dtype=torch.float32)

    assert module.gate_up_proj.requires_grad is False
    assert module.down_proj.requires_grad is False

    hidden_states, top_k_index, top_k_weights = _random_routing(device)
    hidden_states = hidden_states.detach().requires_grad_(True)

    out = module(hidden_states, top_k_index, top_k_weights)
    out.float().sum().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad.float().abs().sum() > 0
    # No gradient ever lands on the frozen storage.
    assert module.gate_up_proj.grad is None
    assert module.down_proj.grad is None


def test_expertsnbit_lora_training_reduces_loss_int8():
    # Same QLoRA-style contract as the 4-bit test, on the 8-bit blockwise base: only the
    # adapters move, the frozen base stays bit-identical.
    torch.manual_seed(0)
    experts_lora = _load_experts_lora()

    gate_up, down = _random_expert_weights(torch.float32, "cpu")
    base = ExpertsNbit.from_float(gate_up, down, quant_type="int8", compute_dtype=torch.float32)
    model = experts_lora(base, r=4, alpha=8)

    gate_up_before = base.gate_up_proj.clone()
    down_before = base.down_proj.clone()

    hidden_states, top_k_index, top_k_weights = _random_routing("cpu")
    target = torch.randn_like(hidden_states)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(hidden_states, top_k_index, top_k_weights), target)
        loss.backward()
        assert base.gate_up_proj.grad is None and base.down_proj.grad is None
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert torch.equal(base.gate_up_proj, gate_up_before)
    assert torch.equal(base.down_proj, down_before)


def test_expertsnbit_quant_type_validation():
    with pytest.raises(ValueError, match="quant_type"):
        ExpertsNbit(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, quant_type="int4")
    # Experts4bit stays a strictly-4-bit specialization: the 8/16-bit schemes are rejected.
    with pytest.raises(ValueError, match="quant_type"):
        Experts4bit(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, quant_type="int8")
    assert issubclass(Experts4bit, ExpertsNbit)


def test_expertsnbit_passthrough_skips_blocksize_check():
    # Passthrough storage has no quantization blocks, so odd dims are fine there ...
    module = ExpertsNbit(NUM_EXPERTS, hidden_dim=100, intermediate_dim=120, quant_type="bf16")
    assert module.gate_up_proj.shape == (NUM_EXPERTS, 2 * 120 * 100)
    # ... but quantized schemes still enforce block alignment.
    with pytest.raises(ValueError, match="divisible by blocksize"):
        ExpertsNbit(NUM_EXPERTS, hidden_dim=100, intermediate_dim=128, quant_type="int8", blocksize=64)


def test_expertsnbit_is_exported():
    assert bnb.nn.ExpertsNbit is ExpertsNbit
