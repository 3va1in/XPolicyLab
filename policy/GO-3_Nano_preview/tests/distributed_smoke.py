"""Run with torchrun --nproc-per-node=2; tiny ZeRO-2 integration, not full SFT."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from go3_nano_preview.sft.distributed import initialize, on_rank_zero, on_all_ranks, mean_losses
from go3_nano_preview.sft.sampler import DistributedWeightedSampler, check_dataset_agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=['gloo','nccl'], default='gloo')
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cpu')
    if args.backend == 'nccl':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda',local_rank)
    control = initialize(6000,backend=args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.output)
    on_rank_zero(lambda: output.mkdir(parents=True,exist_ok=False),control)
    weights = torch.arange(1,65,dtype=torch.float64)
    check_dataset_agreement(weights,control)
    sampler = DistributedWeightedSampler(weights,4,rank=rank,world=world,seed=1016)
    iterator = iter(sampler)
    first = on_all_ranks(lambda: [next(iterator) for _ in range(4)],control)
    pool_list = [None]*world
    dist.all_gather_object(pool_list,sampler.indices.tolist(),group=control)
    assert all(not set(pool_list[i]) & set(pool_list[j]) for i in range(world) for j in range(i))
    means = mean_losses(dict(vgen=float(rank),ae=float(rank+1),cotrain=float(rank+2)),device)
    assert means['vgen'] == (world-1)/2
    def failing_operation():
        raise ValueError('intentional collective failure test')
    for fn in (lambda: on_rank_zero(failing_operation,control),
               lambda: on_all_ranks(failing_operation if rank==1 else lambda: 1,control)):
        try:
            fn()
        except RuntimeError as exc:
            assert 'intentional collective failure test' in str(exc)
        else:
            raise AssertionError('Failure must reach every rank')
    result = dict(world_size=world, backend=args.backend, sampler_disjoint=True,
        global_mean_correct=True,errors_propagated=True,first_batch=first)
    if args.backend == 'nccl':
        import deepspeed
        class TinyCotrain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.video = torch.nn.Linear(4,4)
                self.action = torch.nn.Linear(4,1)
            def forward(self,x,phase):
                if phase == 'vgen':
                    return self.video(x).square().mean()
                if phase == 'ae':
                    return self.action(x).square().mean()
                return self.action(self.video(x)).square().mean()
        torch.manual_seed(42)
        model = TinyCotrain().to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
        ds_config = {'train_micro_batch_size_per_gpu':2,'gradient_accumulation_steps':3,
            'train_batch_size':2*3*world,'zero_optimization':{'stage':2},
            'zero_allow_untested_optimizer':True,'fp16':{'enabled':False},'bf16':{'enabled':True}}
        engine,_,_,_ = deepspeed.initialize(model=model,optimizer=optimizer,config=ds_config,dist_init_required=False)
        initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        x = torch.full((2,4),float(rank+1),device=device,dtype=torch.bfloat16)
        for step in range(3):
            for phase,weight in [('vgen',1.),('ae',0.1),('cotrain',1.)]:
                loss = engine(x,phase)
                engine.backward(loss*weight)
                engine.step()
        trained = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(trained) for _ in range(world)]
        dist.all_gather(gathered,trained)
        assert all(torch.equal(trained,t) for t in gathered)
        assert not torch.equal(initial,trained)
        assert engine.global_steps == 3
        engine.save_checkpoint(str(output/'state'),tag='step3',client_state={'step':3})
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
        loaded,state = engine.load_checkpoint(str(output/'state'),tag='step3')
        assert loaded and state['step']==3
        restored = torch.cat([p.detach().flatten() for p in model.parameters()])
        assert torch.equal(restored,trained)
        timeout = engine.data_parallel_group._get_backend(device).options._timeout
        result.update(zero_stage=2,bf16=True,optimizer_updates=3,rank_parameters_equal=True,
            parameters_changed=True,checkpoint_restore=True,
            process_group_timeout=str(timeout))
        assert timeout.total_seconds() == 6000
    on_rank_zero(lambda: (output/'result.json').write_text(json.dumps(result,indent=2)),control)
    if rank==0: print(json.dumps(result),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
