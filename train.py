import torch
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

from encoder_decoder import EncoderDecoder
from dataset import Dataset
import utils

def train_one_epoch(model, dataloader, optimizer, device, epoch, iterations=1000):
    iterator = iter(utils.cycle(dataloader))

    with tqdm(total=iterations, desc=f'Epoch {epoch + 1}') as pbar:
        loss_lst = []

        for _ in range(iterations):

            mapping = next(iterator)

            optimizer.zero_grad()

            loss, _, _ = model(mapping, device)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.detach().cpu().item()})
            pbar.update(1)

            loss_lst.append(loss.detach().cpu().item())
        
        return loss_lst
    

if __name__ == '__main__':

    arg = argparse.ArgumentParser()

    arg.add_argument('--batch_size', type=int, default=16, help='Batch size of data to train the model.')
    arg.add_argument('--num_iters', type=int, default=1000, help='Number of iterations within each epoch to train the model.')
    arg.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the model.')
    arg.add_argument('--hidden_size', type=int, default=128, help='Size of hidden states encoded by VectorNet.')
    arg.add_argument('--lr0', type=float, default=1e-3, help='Initial learning rate for AdamW to train the model.')
    arg.add_argument('--lrf', type=float, default=1e-4, help='Final learning rate for AdamW to train the model.')
    arg.add_argument('--num_workers', type=int, default=8, help='Number of workers to use for training the model.')

    arg.add_argument('--model_load_path', type=str, default=None, help='Path to load the model (*.pt) file.')
    arg.add_argument('--model_save_path', type=str, default='./models/model.pt', help='Path to save the mode(*.pt) file v.')

    arg.add_argument('--data_dir', type=str, default='./data/train/', help='Path to the training data.')
    arg.add_argument('--core_num', type=int, default=8, help='Number of cores to use for preprocessing the data.')
    arg.add_argument('--load_temp_file', type=bool, default=True, help='Load preprocessed data.')
    arg.add_argument('--temp_file_path', type=str, default="./data/temp/temp_train.pkl", help='Path to the preprocessed data.')

    arg = arg.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder(arg.hidden_size).to(device)
    if arg.model_load_path is not None:
        model.load_state_dict(torch.load(arg.model_load_path))

    dataset = Dataset(arg.data_dir, arg.core_num, arg.temp_file_path, load_temp_file=arg.load_temp_file)
    dataloader = utils.RandomSampler(dataset, arg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=arg.lrf/arg.lr0, total_iters=arg.num_epochs)

    loss_lst = []

    for epoch in range(arg.num_epochs):
        ret = train_one_epoch(model, dataloader, optimizer, device, epoch, iterations=arg.num_iters)
        loss_lst.append(ret)
        scheduler.step()
        print(scheduler.get_last_lr())

    torch.save(model.state_dict(), arg.model_save_path)

    plt.plot(loss_lst)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()

    with open('./model/loss.pkl', 'wb') as f:
        pickle.dump(loss_lst, f)


