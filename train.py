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


def train_one_epoch(model, dataloader, optimizer, epoch, arg, device):
    iterator = iter(utils.cycle(dataloader))

    with tqdm(total=arg.num_iters, desc=f'Epoch {epoch + 1}') as pbar:
        loss_lst = []

        for _ in range(arg.num_iters):

            mapping = next(iterator)

            optimizer.zero_grad()

            loss, _, _ = model(mapping, device=device)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update(1)

            loss_lst.append(loss.detach().cpu().item())
        
        return loss_lst


def single_gpu_training(arg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder(arg.hidden_size, num_workers=arg.core_num).to(device)
    if arg.model_load_path is not None:
        model.load_state_dict(torch.load(arg.model_load_path))

    dataset = Dataset(arg.data_dir, arg.core_num, arg.temp_file_path, load_temp_file=arg.load_temp_file)
    dataloader = utils.RandomSampler(dataset, arg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr0, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=arg.lrf/arg.lr0, total_iters=arg.num_epochs)

    for epoch in range(arg.num_epochs):
        train_one_epoch(model, dataloader, optimizer, epoch, arg, device)
        scheduler.step()

    if not os.path.exists(os.path.dirname(arg.model_save_path)):
        os.makedirs(os.path.dirname(arg.model_save_path), exist_ok=True)

    torch.save(model.state_dict(), arg.model_save_path)


def setup(rank, world_size, is_windows=False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    if is_windows:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_fn(rank, world_size, arg):

    setup(rank, world_size, is_windows=arg.is_windows)

    model = EncoderDecoder(arg.hidden_size, num_workers=arg.core_num).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if arg.model_load_path is not None:
        model.load_state_dict(torch.load(arg.model_load_path))

    dataset = Dataset(arg.data_dir, arg.core_num, arg.temp_file_path, load_temp_file=arg.load_temp_file)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = arg.batch_size // world_size, 
        sampler = sampler,
        collate_fn = lambda batch : [item for item in batch]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr0, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=arg.lrf/arg.lr0, total_iters=arg.num_epochs)

    for epoch in range(arg.num_epochs):
        sampler.set_epoch(epoch) # guarantee a different shuffling order for each epoch

        if rank == 0:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
        else:
            pbar = dataloader
        
        for _, batch in enumerate(pbar):
            optimizer.zero_grad()

            loss, _, _ = model(batch, device=rank)

            loss.backward()
            # print('Norm of gradient:', model.module.decoder.goals_2D_decoder.mlp.linear.weight.grad.norm().item())

            optimizer.step()

            if rank == 0:
                pbar.set_postfix(
                    {
                        'decoder grad norm': model.module.decoder.goals_2D_decoder.mlp.linear.weight.grad.norm().item(), 
                        'loss': loss.detach().cpu().item()
                    }
                )

        scheduler.step()

        dist.barrier() # wait for all processes to finish the current epoch

        if rank == 0:
            if not os.path.exists(os.path.dirname(arg.model_save_path)):
                os.makedirs(os.path.dirname(arg.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), arg.model_save_path)

    dist.destroy_process_group() # clean up


def multi_gpu_training(arg):
    mp.spawn(
        train_fn,
        args=(arg.num_gpus, arg),
        nprocs=arg.num_gpus,
        join=True
    )
    

if __name__ == '__main__':

    arg = argparse.ArgumentParser()

    arg.add_argument('--batch_size', type=int, default=64, help='Batch size of data to train the model.')
    arg.add_argument('--num_iters', type=int, default=100, help='Number of iterations within each epoch to train the model.')
    arg.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model.')
    arg.add_argument('--hidden_size', type=int, default=128, help='Size of hidden states encoded by VectorNet.')
    arg.add_argument('--lr0', type=float, default=1e-3, help='Initial learning rate for AdamW to train the model.')
    arg.add_argument('--lrf', type=float, default=1e-4, help='Final learning rate for AdamW to train the model.')
    arg.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use for training the model.')
    arg.add_argument('--is_windows', action='store_true', help='Set this flag if the OS is Windows.')
    arg.add_argument('--distributed_training', action='store_true', help='Set this flag to train the model in parallel.')

    arg.add_argument('--model_load_path', type=str, default=None, help='Path to load the model (*.pt) file.')
    arg.add_argument('--model_save_path', type=str, default='./models/model.pt', help='Path to save the mode(*.pt) file v.')

    arg.add_argument('--data_dir', type=str, default='./data/train/', help='Path to the training data.')
    arg.add_argument('--core_num', type=int, default=4, help='Number of cores to use for preprocessing the data.')
    arg.add_argument('--load_temp_file', action='store_true', help='Load preprocessed data.')
    arg.add_argument('--temp_file_path', type=str, default="../data/temp_train.pkl", help='Path to the preprocessed data.')

    arg = arg.parse_args()

    assert arg.batch_size % arg.num_gpus == 0 # batch size should be divisible by the number of GPUs

    if arg.distributed_training:
        multi_gpu_training(arg)
    else:
        single_gpu_training(arg)