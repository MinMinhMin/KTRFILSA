import torch

from train import gather_metric_tensors


class _FakeDistributedAccelerator:
    num_processes = 2

    def __init__(self):
        self.pad_calls = []
        self.remote_predictions = torch.tensor([0.3, 0.7, 0.9])
        self.remote_truths = torch.tensor([1.0, 0.0, 1.0])

    def pad_across_processes(self, tensor, dim, pad_index):
        self.pad_calls.append((dim, pad_index, tensor.numel()))
        padding = 3 - tensor.numel()
        if padding <= 0:
            return tensor
        return torch.cat([tensor, tensor.new_full((padding,), pad_index)])

    def gather(self, values):
        assert isinstance(values, tuple)
        predictions, truths = values
        return (
            torch.cat([predictions, self.remote_predictions]),
            torch.cat([truths, self.remote_truths]),
        )


def test_gather_metric_tensors_pads_uneven_ranks_and_removes_padding():
    accelerator = _FakeDistributedAccelerator()
    predictions, truths = gather_metric_tensors(
        accelerator,
        torch.tensor([0.2, 0.8]),
        torch.tensor([0.0, 1.0]),
    )

    assert accelerator.pad_calls == [(0, 0.0, 2), (0, -1.0, 2)]
    assert predictions.tolist() == [0.2, 0.8, 0.3, 0.7, 0.9]
    assert truths.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
