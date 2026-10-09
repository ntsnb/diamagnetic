import io
import pytest
import torch
from paper_impl.backbones import LEGACY_ENCODERS
from paper_impl.model import ModelConfig, PaperForecaster, ForecastBranches
from paper_impl.discrepancy import specialization_loss


@pytest.mark.parametrize("encoder", LEGACY_ENCODERS)
def test_style_backbone_full_forecaster_and_branch_integration(encoder):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = ModelConfig(3, 7, 2, targets=2, hidden=4, heads=1, n_experts=2,
                         latent_block=2, encoder=encoder)
    model, branches = PaperForecaster(config), ForecastBranches(config)
    x = torch.randn(2, 7, 3)
    output = model(x)
    assert output.prediction.shape == (2, 2, 2)
    assert branches(x).shape == (2, 4)
    assert output.reconstruction.shape == x.shape
    attr = model.attribute(x)
    loss = output.prediction.square().mean() + specialization_loss(attr.maps, .3)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = PaperForecaster(config)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x).prediction, output.prediction.detach(), rtol=0, atol=0)
