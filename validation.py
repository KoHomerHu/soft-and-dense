import torch
from tqdm import tqdm
import argparse

from torch.nn.parallel import DataParallel as DP

from encoder_decoder import EncoderDecoder
from dataset import Dataset
import utils
import os

def valid_fn(arg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder(128, num_workers=arg.core_num).to(device)
    model.eval()
    model = DP(model)
    model.load_state_dict(torch.load(arg.model_load_path))

    dataset = Dataset(arg.data_dir, arg.core_num, arg.temp_file_path, load_temp_file=arg.load_temp_file)

    total_fde = 0.0
    total_MR = 0.0
    total_num = 0
    
    pbar = tqdm(dataset, desc='Validating: ')
    
    for _, mapping in enumerate(pbar):
        
        _, scores_lst, dense_goals_lst = model([mapping], device=device)
        _, fde, MR = utils.select_goals_by_optimization(scores_lst, dense_goals_lst, [mapping])

        total_fde += fde.sum()
        total_MR += MR.sum()
        total_num += 1

        pbar.set_postfix({'minFDE': total_fde / total_num, 'MR': total_MR / total_num})

    print(f'Final minFDE: {total_fde / total_num}, Final MR: {total_MR / total_num}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--core_num', type=int, default=7)
    parser.add_argument('--model_load_path', type=str, default='./models/model (3).pt')
    parser.add_argument('--data_dir', type=str, default='./data/train/', help='Path to the training data.')
    parser.add_argument('--load_temp_file', action='store_true', help='Load preprocessed data.')
    parser.add_argument('--temp_file_path', type=str, default="./data/temp/temp_train.pkl", help='Path to the preprocessed data.')

    arg = parser.parse_args()

    valid_fn(arg)

