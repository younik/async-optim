from abc import ABC, abstractmethod
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Union

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import allreduce_average, pack, unpack


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass


class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        buffer, shapes = pack(gradients)
        allreduce_average(buffer)
        out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100,
    async_error: bool = False
    cut: int = 1


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
                async_error=config.async_error,
                cut = config.cut
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor], timing=None) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._allreduce.aggregate(gradients)

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate(compressed_grads, timing=timing),
            self._allreduce.aggregate(uncompressed_grads),
        )

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    async_error: bool = False
    cut: int = 1

class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)

    def aggregate(self, gradients: List[torch.Tensor], timing=None) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contaiasync_errorn the 'approximation error', used for the error feedback
        mechanism.
        """

        # Un-batch the approximation and error feedback, write to the output        
        if not hasattr(self, 'error'):
            total_size = sum([g.numel() for g in gradients])
            self.error = torch.zeros(total_size, device=self.device)
            self.reduce_error = torch.zeros(total_size, device=self.device)

            self.start_window_pos = 0

            window_size = self.error.numel()
            for _ in range(self.config.cut - 1):
                print("Reducing by half window size of", window_size)
                window_size = - (window_size // -2) # ceil division
            
            end_pos = window_size
            self.error_window = self.error[self.start_window_pos : end_pos]
            self.reduce_window = self.reduce_error[self.start_window_pos : end_pos]
        else:
            window_size = self.reduce_window.numel()
            self.start_window_pos += self.reduce_window.numel()
            if self.start_window_pos >= self.error.numel():
                self.start_window_pos = 0
            end_pos = min(self.start_window_pos + window_size, self.error.numel())
            
            self.error_window = self.error[self.start_window_pos : end_pos]
            self.reduce_window = self.reduce_error[self.start_window_pos : end_pos]

            self.reduce_handler.wait()

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        # add error feedback (inefficient, mb can coincide with self.error)
        start_idx = 0
        for group in shape_groups:
            for mb in group["grad_batch"]:
                mb.view(-1).add_(self.error[start_idx : start_idx + mb.numel()])
                start_idx += mb.numel()

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = lambda g: g.permute([0, 2, 1])
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                out_batch[:] = torch.einsum(
                    "bmn, bmr -> bnr",
                    maybe_transpose(group["grad_batch"]),
                    in_batch,
                )

            # Average across workers
            allreduce_average(out_buffer)

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                iter_approx = torch.einsum("bnr, bmr -> bnm", in_batch, out_batch)
                maybe_transpose(group["grad_batch"]).sub_(iter_approx)  # error feedback
                maybe_transpose(group["approximation"]).add_(iter_approx)
                del iter_approx
        
        start_idx = 0
        for group in shape_groups:
            for g, approx, mb in zip(
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                g[:] = approx + self.reduce_error[start_idx : start_idx + mb.numel()].view_as(g)
                self.error[start_idx : start_idx + mb.numel()] = mb.view(-1)
                start_idx += mb.numel()

        timing[0].record() #
        if self.config.async_error:
            self.reduce_error.zero_()
            self.reduce_window[:] = self.error_window
            self.error_window.zero_()
            self.reduce_handler = allreduce_average(self.reduce_window, async_op=True)
        else:
            self.reduce_handler = SimpleNamespace(wait = lambda: None, is_completed = lambda: True)
        timing[1].record() #

        # Increment the step counter
        self.step_counter += 1

        return gradients

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)
