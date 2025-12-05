# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Full iteration CUDA graph for training."""

import logging

import torch

from megatron.core.tensor_parallel.random import get_all_rng_states
from megatron.core.pipeline_parallel.moe_packed_offload import (
    packed_moe_expert_offloading_reset,
)

logger = logging.getLogger(__name__)

# The below functions traverse through nested data structures (tuples, lists, dicts)
# present in src and creates a deep copy where all PyTorch tensors are cloned,
# detached from the computation graph, and moved to CUDA device. Non-tensor objects
# are returned as-is.


def copy_tensors_in_struct(src):
    """Copy src to new tensors."""
    if isinstance(src, tuple):
        return tuple(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, list):
        return list(copy_tensors_in_struct(i) for i in src)
    elif isinstance(src, dict):
        return {k: copy_tensors_in_struct(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def clone_tensors_in_struct(tgt, src):
    """Copy src to pre-existing tensors in tgt."""
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                clone_tensors_in_struct(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")


# Class to copy dataloader output to static CUDA tensors for CUDA graph input. This
# maintains separate static buffers for training and validation CUDA graphs.
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(copy_tensors_in_struct(inputs))
        else:

            for k in inputs.keys():
                if k not in StaticBufferLoader.static_buffers[stage][microbatch]:
                    if isinstance(inputs[k], torch.Tensor):
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = torch.empty_like(
                            inputs[k], device="cuda"
                        )
                    else:
                        StaticBufferLoader.static_buffers[stage][microbatch][k] = inputs[k]

            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch], inputs
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]


class FullCudaGraphWrapper:
    """Wrapper class to enable FullIterationCUDAgraph."""

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}

    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1, packed_moe_expert_offloading=False):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        self.packed_moe_expert_offloading = packed_moe_expert_offloading

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs from Dataloader and copy to static buffers."""
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            if iterator0 is not None:
                for b in range(num_microbatches):
                    data_list.append(
                        self.static_loader(
                            next(iterator0), 'training' if training else 'validation', b
                        )
                    )
                data_list = [iter(data_list)]
            else:
                data_list.append(None)
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                if data_iterator[i] is not None:
                    data_list_i = []
                    for b in range(num_microbatches):
                        data_list_i.append(
                            self.static_loader(
                                next(data_iterator[i]), 'training' if training else 'validation', b
                            )
                        )
                    data_list.append(iter(data_list_i))
                else:
                    data_list.append(None)
        return data_list

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, 'forward_backward_func does not accept positional args'
        assert all(
            [
                kwarg in kwargs
                for kwarg in [
                    'model',
                    'data_iterator',
                    'num_microbatches',
                    'seq_length',
                    'forward_only',
                ]
            ]
        )
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else 'validation'
        curr_iteration = self.curr_iter(training_str)
        if curr_iteration == self.cuda_graph_warmup_steps:
            print(f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert FullCudaGraphWrapper.cuda_graph[training_str] is None
            FullCudaGraphWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCudaGraphWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(
                FullCudaGraphWrapper.cuda_graph[training_str],
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(
                    *args, **kwargs
                )
            torch.cuda.synchronize()
            torch.distributed.barrier()
            logger.info(f'CUDA graph capture done!!!')

        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
        else:
            # packed_moe_expert_offloading_reset(enabled=self.packed_moe_expert_offloading and training)
            FullCudaGraphWrapper.cuda_graph[training_str].replay()
        list_max0 = []
        list_max1 = []
        list_max2 = []
        list_max3 = []
        list_num_tokens = []
        # Flag to track if condition is met on any rank
        condition_met = torch.zeros(1, dtype=torch.bool, device='cuda')
        
        for model_chunk in model:
            for layer in model_chunk.module.module.decoder.layers:
                mlp = layer.mlp
                if hasattr(mlp, 'token_dispatcher') and hasattr(mlp.token_dispatcher, '_comm_manager'):
                    for i, x in enumerate(mlp.token_dispatcher._comm_manager.list_record_fwd):
                        num_tokens = mlp.token_dispatcher._comm_manager.list_record_m[i].sum().item()
                        list_num_tokens.append(num_tokens)
                        if num_tokens > 0:
                            list_max0.append(mlp.token_dispatcher._comm_manager.list_record_fwd[i][0][:num_tokens].abs().max().item())
                            list_max1.append(mlp.token_dispatcher._comm_manager.list_record_fwd[i][1].abs().max().item())
                            list_max2.append(mlp.token_dispatcher._comm_manager.list_record_bwd[i][0].abs().max().item())
                            list_max3.append(mlp.token_dispatcher._comm_manager.list_record_bwd[i][1][:num_tokens].abs().max().item())
                            if torch.distributed.get_rank() == 15 and mlp.token_dispatcher._comm_manager.list_record_bwd[i][1][:num_tokens].abs().max().item() > 0.01:
                                condition_met.fill_(True)
        
        # Allreduce to notify all GPUs if condition was met on any rank
        torch.distributed.all_reduce(condition_met, op=torch.distributed.ReduceOp.MAX)
        
        if condition_met.item():
            # Collect all tensors for debugging
            debug_data = {
                'list_max0': list_max0,
                'list_max1': list_max1,
                'list_max2': list_max2,
                'list_max3': list_max3,
                'list_num_tokens': list_num_tokens,
                'model_chunks': []
            }
            
            for model_chunk_i, model_chunk in enumerate(model):
                chunk_data = {'chunk_id': model_chunk_i, 'layers': []}
                for layer_i, layer in enumerate(model_chunk.module.module.decoder.layers):
                    mlp = layer.mlp
                    if hasattr(mlp, 'token_dispatcher') and hasattr(mlp.token_dispatcher, '_comm_manager'):
                        comm_manager = mlp.token_dispatcher._comm_manager
                        layer_data = {
                            'layer_id': layer_i,
                            'records': []
                        }
                        
                        for i, x in enumerate(comm_manager.list_record_fwd):
                            num_tokens = comm_manager.list_record_m[i].sum().item()
                            record = {
                                'record_id': i,
                                'num_tokens': num_tokens,
                                'list_record_fwd': [t.detach().cpu() for t in comm_manager.list_record_fwd[i]],
                                'list_record_bwd': [t.detach().cpu() for t in comm_manager.list_record_bwd[i]],
                                'list_record_m': comm_manager.list_record_m[i].detach().cpu(),
                            }
                            layer_data['records'].append(record)
                        
                        chunk_data['layers'].append(layer_data)
                debug_data['model_chunks'].append(chunk_data)
            
            # Save to file
            rank = torch.distributed.get_rank()
            filename = f'/lustre/fsw/coreai_mlperf_training/users/nanz/moe/megatron-moe-scripts/results/debug_tensors_rank{rank}_iter{FullCudaGraphWrapper.curr_iteration[training_str]}.pt'
            torch.save(debug_data, filename)
            logger.info(f'Rank {rank}: Saved debug tensors to {filename}')
            print(f'Rank {rank}: Saved debug tensors to {filename}', flush=True)
            

        # if torch.distributed.get_rank() == 0: import pdb; pdb.set_trace()
        # print(f"Rank {torch.distributed.get_rank()}: num_tokens {list_num_tokens}", flush=True)
        # print(f"Rank {torch.distributed.get_rank()}: fwd 0 {list_max0}", flush=True)
        # print(f"Rank {torch.distributed.get_rank()}: fwd 1 {list_max1}", flush=True)
        # print(f"Rank {torch.distributed.get_rank()}: bwd 0 {list_max2}", flush=True)
        # print(f"Rank {torch.distributed.get_rank()}: bwd 1 {list_max3}", flush=True)
        if FullCudaGraphWrapper.cuda_graph[training_str] is None:
            if curr_iteration < self.cuda_graph_warmup_steps - 1:
                for model_chunk in model:
                    for layer in model_chunk.module.module.decoder.layers:
                        mlp = layer.mlp
                        if hasattr(mlp, 'token_dispatcher') and hasattr(mlp.token_dispatcher, '_comm_manager'):
                            mlp.token_dispatcher._comm_manager.list_record_fwd = []
                            mlp.token_dispatcher._comm_manager.list_record_bwd = []
                            mlp.token_dispatcher._comm_manager.list_record_m = []
                            mlp.token_dispatcher._comm_manager.index_fwd = 0
                            mlp.token_dispatcher._comm_manager.index_bwd = 0
                            mlp.token_dispatcher._comm_manager.index_fwd_g = [0]
                            mlp.token_dispatcher._comm_manager.index_bwd_g = [0]

            # if torch.distributed.get_rank() == 0: import pdb; pdb.set_trace()
        self.speculative_cuda_graph_check(model)
        self.next_iter(training_str)
        return FullCudaGraphWrapper.result[training_str]

    def speculative_cuda_graph_check(self, model):
        ''' check speculative execution modules '''
        if self.packed_moe_expert_offloading is not None:
            # Check if there is any overflow in the receiving buffer
            over_budget = torch.zeros(1, dtype=torch.bool, device='cuda')
            for model_chunk in model:
                for layer in model_chunk.module.module.decoder.layers:
                    mlp = layer.mlp
                    if hasattr(mlp, 'token_dispatcher') and hasattr(mlp.token_dispatcher, 'check_over_budget'):
                        over_budget |= mlp.token_dispatcher.check_over_budget()
            if over_budget.item():
                raise Exception(f"Rank {torch.distributed.get_rank()} overbudget")

    def curr_iter(self, stage):
        """Return current training/validation iteration."""
        return FullCudaGraphWrapper.curr_iteration[stage]

    def next_iter(self, stage):
        """Increment current training/validation iteration."""
        FullCudaGraphWrapper.curr_iteration[stage] += 1
