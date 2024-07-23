import torch

from graphlinear import Linear, assert_no_graph_break

def test_f_no_graph_break() -> None:
    m = Linear("0e + 1e + 2e", "0e + 2x1e + 2e", f_in=44, f_out=25, _optimize_einsums=False)
    assert_no_graph_break(m,torch.randn(10, 44, 9))
