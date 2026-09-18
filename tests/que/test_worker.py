from src.que.worker import _is_wandb_injected_stop


class TestIsWandbInjectedStop:
    def test_bare_exception_is_detected(self) -> None:
        assert _is_wandb_injected_stop(Exception())

    def test_exception_with_message_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(Exception("boom"))

    def test_exception_subclass_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(RuntimeError())

    def test_unrelated_error_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(ValueError("bad input"))
