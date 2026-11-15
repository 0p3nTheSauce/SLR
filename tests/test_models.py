import pytest
import torch

from src.models import S3D, MViTv2_S_e, get_model
from src.models.pytorch_mvit import MViTv2S_extended


def test_random_init_mvitv2_s_e_skips_the_k400_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("pretrained weights were requested")

    monkeypatch.setattr("torchvision.models._api.WeightsEnum.get_state_dict", no_download)
    model = get_model(MViTv2_S_e, num_classes=10, drop_p=0.1, pretrained=False)
    assert isinstance(model, MViTv2S_extended)
    assert model(torch.zeros(1, 3, 32, 224, 224)).shape == (1, 10)


def test_random_init_rejected_for_models_without_the_option() -> None:
    with pytest.raises(ValueError, match="without pretrained weights"):
        get_model(S3D, num_classes=10, drop_p=0.1, pretrained=False)
