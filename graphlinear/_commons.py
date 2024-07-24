import torch
from typing import Optional, List, Any, Tuple, Callable
from torch._dynamo import reset
from torch._dynamo.eval_frame import optimize


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def get_graph_breaks(f: torch.nn.Module, *extra_args, **extra_kwargs) -> Tuple[int, int]:
    """

    Runs TorchDynamo on the supplied module and returns the total graph break count, if any.

    Parameters
    ----------
        f: torch.nn.Module
        extra_args: Any
                    Inputs to f
        extra_kwargs: Any
                    Inputs to f

    Returns
    -------
        (graph_count, graph_break_count): Tuple[int, int]

    """

    def inner(*args, **kwargs):

        reset()

        graphs: List[torch.fx.GraphModule] = []
        break_reasons: List[Any] = []
        op_count: int = 0
        ops_per_graph: List[torch.fx.Node] = []
        out_guards: List[_guards.Guard] = []

        def dynamo_graph_accumulating_compiler(
                gm: torch.fx.GraphModule, example_inputs
        ):
            from torch._dynamo.backends.debugging import _explain_graph_detail

            nonlocal graphs
            nonlocal op_count
            nonlocal ops_per_graph
            nonlocal break_reasons

            gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(
                gm, graphs, op_count, ops_per_graph, break_reasons
            )

            return gm.forward

        def guard_export_print(guards):
            nonlocal out_guards
            out_guards.extend(guards)

        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)

        opt_f(*args, **kwargs)

        graph_count = len(graphs)
        graph_break_count = graph_count - 1


        reset()

        return (graph_count, graph_break_count)

    if extra_args or extra_kwargs:
        warnings.warn(
            "explain(f, *args, **kwargs) is deprecated, use explain(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your explain call in the future if your user defined kwargs "
            "conflict with future kwargs added to explain(f).",
            FutureWarning,
            stacklevel=2,
        )
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner


def assert_no_graph_break(
        model: Optional[Callable] = None,
        *extra_args, **extra_kwargs,
) -> None:
    """
    Given a model/function using TorchDynamo and a specified backend, check for graph breaks.

    Parameters
    ----------
        model: Callable
            Module/function to identify for graph breaks during compilation
        *extra_args, **extra_kwargs : Any
            Model inputs

    Returns
    _______
        None
    """
    graph_count, graph_break_count = get_graph_breaks(model)(*extra_args, **extra_kwargs)
    assert graph_break_count == 0, (repr(model) + " Is resulting to " +
                                    str(graph_count) + "graphs, with " +
                                    str(graph_break_count) + " graph break")

