import torch
from tqdm import tqdm
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from encoder_decoder import EncoderDecoder
from dataset import Dataset
import utils
import os

def setup(rank, world_size, is_windows=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    if is_windows:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def valid_fn(rank, world_size, arg):

    setup(rank, world_size, is_windows=arg.is_windows)

    model = EncoderDecoder(128, num_workers=arg.core_num).to(rank)
    model.eval()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.load_state_dict(torch.load(arg.model_load_path))

    dataset = Dataset(arg.data_dir, arg.core_num, arg.temp_file_path, load_temp_file=arg.load_temp_file)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = arg.batch_size // world_size, 
        sampler = sampler,
        collate_fn = lambda batch : [item for item in batch]
    )

    total_fde = 0.0
    total_MR = 0.0
    total_num = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Validating: ')
    else:
        pbar = dataloader
    
    for _, batch in enumerate(pbar):

        _, scores_lst, dense_goals_lst = model(batch, device=rank)
        _, fde, MR = utils.select_goals_by_optimization(scores_lst, dense_goals_lst, batch)

        total_fde += fde.sum()
        total_MR += MR.sum()
        total_num += len(batch)

        dist.barrier() # wait for all processes to finish the current epoch

        if rank == 0:
            pbar.set_postfix({'minFDE': total_fde / total_num, 'MR': total_MR / total_num})

    total_fde_lst[rank] = total_fde
    total_MR_lst[rank] = total_MR

    dist.destroy_process_group() # clean up

def multi_gpu_training(arg):
    global total_fde_lst, total_MR_lst

    mp.spawn(
        valid_fn,
        args=(arg.num_gpus, arg),
        nprocs=arg.num_gpus,
        join=True
    )

    print('minFDE:', sum(total_fde_lst) / len(total_fde_lst))
    print('MR:', sum(total_MR_lst) / len(total_MR_lst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--core_num', type=int, default=7)
    parser.add_argument('--model_load_path', type=str, default='./models/model (3).pt')
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--is_windows', action='store_true')

    parser.add_argument('--data_dir', type=str, default='./data/train/', help='Path to the training data.')
    parser.add_argument('--load_temp_file', action='store_true', help='Load preprocessed data.')
    parser.add_argument('--temp_file_path', type=str, default="./data/temp/temp_train.pkl", help='Path to the preprocessed data.')

    arg = parser.parse_args()

    assert arg.batch_size % arg.num_gpus == 0 # batch size should be divisible by the number of GPUs

    total_fde_lst = [None for _ in range(arg.num_gpus)]
    total_MR_lst = [None for _ in range(arg.num_gpus)]

    multi_gpu_training(arg)
