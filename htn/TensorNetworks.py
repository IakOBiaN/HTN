import gc

import numpy as np


def identity(dimensions, elements):
    """Return a generalised Kronecker delta ("copy") tensor.

    The tensor has shape ``(elements,) * dimensions`` and contains 1 at
    positions where all indices are equal, and 0 everywhere else.  Used as
    a copy node in tensor-network diagrams.

    Parameters
    ----------
    dimensions : int
            Rank (number of legs) of the tensor.
    elements : int
            Size of each dimension.
    """
    id = np.zeros((elements,) * dimensions)
    for i in range(elements):
        id[((i,) * dimensions)] = 1
    return id


def htn_step(tensors, scale, norm, calc):
    """Perform one HTN renormalization step on the input tensor.

    Contracts the current tensor with its replicas according to the
    hierarchical generator of the chosen lattice (``calc.lattice``),
    producing the next-scale effective tensor.  Also tracks the running
    log-norm (``scale``) so that the partition function does not overflow
    and updates ``calc.nodes`` with the number of physical sites
    represented by the current tensor.

    Parameters
    ----------
    tensors : list of numpy.ndarray
            Single-element list holding the current effective tensor; updated
            in place.
    scale : float
            Accumulated logarithm of normalisation factors from previous
            iterations.
    norm : float
            Norm from the previous iteration (unused as input; preserved for
            API stability).
    calc : htn.MainScripts.CalcConfig
            Configuration; ``calc.lattice``, ``calc.metParam`` and
            ``calc.scale`` are read; ``calc.nodes`` is written.

    Returns
    -------
    tuple
            ``(tensors, scale, norm)`` after the step.
    """
    if calc.lattice == "FSHL":
        size = calc.metParam
        edges_in = (1 + size * 2) ** 2
        edges = edges_in
        nodes = 2 + (size * 2) * (size + 1)
        calc.nodes = nodes
        for i in range(calc.scale):
            calc.nodes += edges * (nodes - 2)
            edges *= edges_in
            # print(calc.nodes, edges)
        tensor = tensors[0]
        norm = tensor.max()
        if norm != 0:
            tensor /= norm
            scale += np.log(norm)
            scale *= edges_in

        cd3 = identity(3, tensor.shape[0])

        dop_tensor = identity(size + 2, tensor.shape[0])
        dop_tensor_2 = identity(size + 2, tensor.shape[0])

        doubled_tensor = np.einsum("ij, aic -> ajc", tensor, cd3)
        doubled_tensor = np.einsum("ij, abi -> abj", tensor, doubled_tensor)
        doubled_tensor = np.einsum("ijk, ajk -> ai", doubled_tensor, cd3)

        for _ in range(size):
            dop_tensor = np.tensordot(dop_tensor, doubled_tensor, axes=([1], [0]))
        dop_tensor = np.tensordot(dop_tensor, tensor, axes=([1], [0]))

        for _ in range(size + (size - 1)):
            dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-1], [0]))
            for j in range(size):
                dop_tensor = np.tensordot(dop_tensor, cd3, axes=([-2 - j], [0]))
                dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-2], [0]))
                dop_tensor = np.tensordot(dop_tensor, cd3, axes=([-1, -3], [0, 1]))
                dop_tensor = np.tensordot(dop_tensor, tensor, axes=([-2], [0]))

        for _ in range(size):
            dop_tensor = np.tensordot(dop_tensor, doubled_tensor, axes=([1], [0]))
        dop_tensor = np.tensordot(dop_tensor, tensor, axes=([1], [0]))
        tensor = np.tensordot(
            dop_tensor, dop_tensor_2, axes=(np.arange(1, size + 2, 1), np.arange(1, size + 2, 1))
        )
    elif calc.lattice == "diamond":
        edges_in = 4
        edges = edges_in
        nodes = 4
        calc.nodes = nodes
        for i in range(calc.scale):
            calc.nodes += edges * (nodes - 2)
            edges *= edges_in
        tensor = tensors[0]
        norm = tensor.max()
        if norm != 0:
            tensor /= norm
            scale += np.log(norm)
            scale *= edges_in

        cd3 = identity(3, tensor.shape[0])
        tensor_dop = np.einsum("ia, abc -> ibc", tensor, cd3)
        tensor_dop = np.einsum("ia, abc -> ibc", tensor, tensor_dop)
        tensor_dop = np.einsum("ib, abc -> aic", tensor, tensor_dop)
        tensor_dop = np.einsum("ib, abc -> aic", tensor, tensor_dop)
        tensor = np.einsum("ijk, jkc -> ic", cd3, tensor_dop)

    tensors[0] = tensor
    norm = np.einsum("ij -> ", tensor)
    gc.collect()
    return (tensors, scale, norm)
