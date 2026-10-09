"""Default-width/depth paths that tiny official model tests do not reach."""
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from paper_impl.model import ModelConfig, PaperForecaster
from paper_impl.official_lrp import (_functionalize_linear_bias_additions,
                                    UnsupportedLRPOperator, _trace)
from paper_impl.discrepancy import specialization_loss


def test_isolated_linear_bias_canonicalization_preserves_value_and_gradients():
    x = torch.randn(2, 3, dtype=torch.double)
    weight = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    bias = torch.randn(4, dtype=torch.double, requires_grad=True)
    graph = make_fx(lambda value: (value @ weight).view(2, 4).add_(bias))(x)
    assert any(n.target == torch.ops.aten.add_.Tensor for n in graph.graph.nodes)
    _functionalize_linear_bias_additions(graph)
    actual, expected = graph(x), x @ weight + bias
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (weight, bias))
    ge = torch.autograd.grad(expected.square().sum(), (weight, bias))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_linear_canonicalization_rejects_observable_alias_and_input_mutation():
    x = torch.ones(2, 3)
    weight = torch.ones(3, 4)
    def aliased(value):
        base = value @ weight
        view = base.view(2, 4)
        view.add_(1.)
        return base + view
    for fn in (aliased, lambda value: value.add_(1.)):
        graph = make_fx(fn)(x.clone())
        with pytest.raises(UnsupportedLRPOperator, match="aliased|isolated"):
            _functionalize_linear_bias_additions(graph)


def test_crossformer_default_dimensions_full_relevance_and_parameter_path():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(71)
        cfg = ModelConfig(8, 24, 12, targets=1, target_indices=(0,), n_experts=3,
                          hidden=64, heads=8, depth=2, expert_backbone="crossformer",
                          official_options={"crossformer_seg_len": 6, "dropout": .1})
        model = PaperForecaster(cfg)
        x = torch.randn(1, 24, 8)
        model.eval()
        graph = _trace(model.experts[0], x, True, 42)
        assert any(n.meta.get("functionalized_linear_bias") for n in graph.graph.nodes)
        torch.testing.assert_close(graph(x), model.experts[0](x), rtol=1e-5, atol=1e-6)
        attr = model.attribute(x, freeze_head=True)
        assert attr.maps.shape == (1, 1, 3, 8, 24)
        assert torch.isfinite(attr.maps).all()
        assert attr.conservation_error.max() < 1e-4
        specialization_loss(attr.maps, .99).backward()
        head_ids = {id(p) for e in model.experts for p in e.head_parameters()}
        assert all(p.grad is None for e in model.experts for p in e.head_parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in model.experts.parameters() if id(p) not in head_ids)
    finally:
        torch.set_num_threads(previous)
