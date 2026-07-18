import torch
from torch import nn
import yaml
import os
from syn_DI_dataset import make_dataset, make_dataloader
from utils import collate_fn_padd, hpe_train
from evaluate import error
from X_Fi import X_Fi
import argparse

def main():
    parser = argparse.ArgumentParser('X-Fi model for MMFi HPE')
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset, e.g. d:/Data/My_MMFi_Data/MMFi_Dataset')
    parser.add_argument('--resume', action='store_true', help='explicitly resume from ./pre-trained_weights/last.pth')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to config yaml')
    parser.add_argument('--save-dir', type=str, default='./pre-trained_weights', help='checkpoint output directory')
    parser.add_argument('--epochs', type=int, default=None, help='override training epochs')
    parser.add_argument('--max-train-batches', type=int, default=1000, help='maximum train batches per epoch')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Available training resources:{device}') 
    resume_requested = args.resume or os.getenv('ORI_XFI_RESUME', '0') == '1'
    print(f'[Ori-HPE] Resume requested: {resume_requested}')

    # load config
    with open(args.config, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    if args.epochs is not None:
        config['training_epochs'] = int(args.epochs)
        
    # load dataset and dataloader
    train_dataset, val_dataset = make_dataset(args.dataset, config)
    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'], collate_fn = collate_fn_padd)

    # load model
    torch.manual_seed(3407)
    model = X_Fi()
    model.to(device)

    # Train the model
    train_criterion = nn.MSELoss()
    test_criterion = error
    hpe_train(
        model=model,
        train_loader = train_loader,
        test_loader = val_loader,
        num_epochs = config['training_epochs'],
        learning_rate = config['learning_rate'],
        train_criterion = train_criterion,
        test_criterion = test_criterion,
        device = device,
        save_dir = args.save_dir,
        val_random_seed = config['modality_existances']['val_random_seed'],
        resume_path = os.path.join(args.save_dir, 'last.pth') if resume_requested else None,
        max_train_batches = args.max_train_batches,
    )
    return
    
if __name__ == '__main__':
    main()
