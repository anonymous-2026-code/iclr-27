"""Two unchanged preflight checks used by the original method extension."""
import torch
from section5_repro.model import WindowAutoencoder
from section5_repro.sana import SANA, SANAPipeline
from section5_repro.supplement_protocol import e1_configuration_unavailable, load_supplement_protocol, variants_for

def test_final_supplement_variant_grids() -> None:
    protocol = load_supplement_protocol()
    e1_scheduled = sum(
        len(variants_for(protocol, "e1", dataset, seed))
        for dataset in protocol.datasets
        for seed in protocol.seeds
    )
    e1_target = sum(
        len(
            variants_for(
                protocol,
                "e1",
                dataset,
                seed,
                include_known_unavailable=True,
            )
        )
        for dataset in protocol.datasets
        for seed in protocol.seeds
    )
    assert e1_scheduled == 276
    assert e1_target == 318
    assert e1_configuration_unavailable(
        protocol, "MSL", 42, "sgd_ttlr0.001"
    )
    assert not e1_configuration_unavailable(
        protocol, "MSL", 42, "adamw_ttlr0.0001"
    )
    assert sum(
        len(variants_for(protocol, "e2", dataset, seed))
        for dataset in protocol.datasets
        for seed in protocol.seeds
    ) == 45 * 7
    assert sum(
        len(variants_for(protocol, "e3", dataset, seed))
        for dataset in protocol.datasets
        for seed in protocol.seeds
    ) == 4 * 3 * 2
    assert sum(
        len(variants_for(protocol, "e6", dataset, seed))
        for dataset in protocol.datasets
        for seed in protocol.seeds
    ) == 45 * 6
    assert sum(
        len(variants_for(protocol, "e7", dataset, seed))
        for dataset in protocol.datasets
        for seed in protocol.seeds
    ) == 45 * 4
    assert protocol.e2.bottom_k == 11

def test_vectorized_sana_matches_reference_and_zero_gate_is_identity() -> None:
    torch.manual_seed(3)
    sana = SANA(variables=3, window=12, d_model=16, heads=4, d_ff=16)
    with torch.no_grad():
        sana.gating.fill_(0.2)
    value = torch.randn(2, 12, 3)
    torch.testing.assert_close(
        sana(value), sana.forward_reference(value), rtol=1e-5, atol=1e-7
    )
    backbone = WindowAutoencoder(12, 3)
    pipeline = SANAPipeline(backbone, variables=3, window=12)
    with torch.inference_mode():
        torch.testing.assert_close(
            pipeline(value),
            backbone(value),
            rtol=0.0,
            atol=0.0,
        )
