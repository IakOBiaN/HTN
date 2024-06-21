import numpy as np
import gc

def identity(dimensions, elements):
	id = np.zeros((elements, ) * dimensions)
	for i in range(elements):
		id[((i, ) * dimensions)] = 1
	return id

def htn_step(tensors, scale, norm, calc):
	if calc.lattice == "FSHL":
		size = calc.metParam
		edges_in = (1 + size * 2) ** 2
		edges = edges_in
		nodes =  2 + (size * 2) * (size + 1)
		calc.nodes = nodes
		for i in range(calc.scale):
			calc.nodes += edges * (nodes - 2)
			edges *= edges_in
			#print(calc.nodes, edges)
		tensor = tensors[0]
		norm = tensor.max()
		if norm != 0:
			tensor /= norm
			scale += np.log(norm)
			scale *= edges_in

		cd3 = identity(3, tensor.shape[0])
		cd4 = identity(4, tensor.shape[0])

		dop_tensor = identity(size + 2, tensor.shape[0])
		dop_tensor_2 = identity(size + 2, tensor.shape[0])

		doubled_tensor = np.einsum("ij, aic -> ajc", tensor, cd3)
		doubled_tensor = np.einsum("ij, abi -> abj", tensor, doubled_tensor)
		doubled_tensor = np.einsum("ijk, ajk -> ai", doubled_tensor, cd3)

		for _ in range(size):
			dop_tensor = np.tensordot(dop_tensor, doubled_tensor, axes = ([1], [0]))
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
		tensor = np.tensordot(dop_tensor, dop_tensor_2, axes = (np.arange(1, size + 2, 1), np.arange(1, size + 2, 1)))
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
