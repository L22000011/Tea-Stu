import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

def generate_none_empth_modality_list():
    modality_list = random.choices(
        [True, False],
        k= 3,
        weights=[50, 50]
    )
    lidar_ = random.choices(
        [True, False],
        k= 1,
        weights=[70, 30]
    )
    wifi_ = random.choices(
        [True, False],
        k= 1,
        weights=[70, 30]
    )
    final_list = modality_list + lidar_ + wifi_
    if sum(final_list) == 0:
        final_list = generate_none_empth_modality_list()
        return final_list
    else:
        return final_list
    
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    kpts = []
    [kpts.append(np.array(t['output'])) for t in batch]
    kpts = torch.FloatTensor(np.array(kpts))

    lengths = torch.tensor([t['input_mmwave'].shape[0] for t in batch ])

    # vk
    VK_data = np.array([(t['input_vk']) for t in batch ])
    VK_data = torch.FloatTensor(VK_data)

    # depth
    depth_data = np.array([(t['input_depth']) for t in batch ])
    depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)

    # mmwave
    ## padd
    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
    ## compute mask
    mmwave_data = mmwave_data.permute(1,0,2)

    # lidar
    ## padd
    lidar_data = [torch.Tensor(t['input_lidar']) for t in batch ]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
    lidar_data = lidar_data.permute(1,0,2)

    # wifi-csi
    wifi_data = np.array([(t['input_wifi-csi']) for t in batch ])
    wifi_data = torch.FloatTensor(wifi_data)
    
    modality_list = generate_none_empth_modality_list()

    return VK_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list

def hpe_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):
    model.eval()
    test_mpjpe = 0
    test_pampjpe = 0
    test_mse = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        VK_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list = data
        VK_data = VK_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        kpts.to(device)
        labels = kpts.type(torch.FloatTensor)
        outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, modality_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        test_mse += criterion1(outputs,labels).item() * VK_data.size(0)

        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        
        mpjpe, pampjpe = criterion2(outputs,labels)
        test_mpjpe += mpjpe.item() * VK_data.size(0)
        test_pampjpe += pampjpe.item() * VK_data.size(0)
    test_mpjpe = test_mpjpe/len(tensor_loader.dataset)
    test_pampjpe = test_pampjpe/len(tensor_loader.dataset)
    test_mse = test_mse/len(tensor_loader.dataset)
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(float(test_mse), float(test_mpjpe),float(test_pampjpe)))
    return test_mpjpe

def hpe_train(model, train_loader, test_loader, num_epochs, learning_rate, train_criterion, test_criterion, device, save_dir, val_random_seed):
    optimizer = torch.optim.AdamW(
        [
                {'params': model.feature_extractor.vk_extractor.parameters()},
                {'params': model.linear_projector.parameters()},
                {'params': model.X_Fusion_block.parameters()}
            ],
        lr = learning_rate
    )
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_mpjpe = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        random.seed(epoch)
        num_iter = 1000
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                VK_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, modality_list = data
                VK_data = VK_data.to(device)
                depth_data = depth_data.to(device)
                lidar_data = lidar_data.to(device)
                mmwave_data = mmwave_data.to(device)
                wifi_data = wifi_data.to(device)
                labels = kpts.to(device)
                labels = labels.type(torch.FloatTensor)
                
                optimizer.zero_grad()
                outputs = model(VK_data, depth_data,  mmwave_data, lidar_data,wifi_data, modality_list)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = train_criterion(outputs,labels)
                if loss == float('nan'):
                    print('nan')
                    print(outputs)
                    print(labels)
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * VK_data.size(0)
            else:
                break
        epoch_loss = epoch_loss/(VK_data.size(0)*num_iter)
        print('Epoch: {}, Loss: {:.8f}'.format(epoch, epoch_loss))
        if (epoch+1) % 10 == 0:
            test_mpjpe = hpe_test(
                model=model,
                tensor_loader=test_loader,
                criterion1 = train_criterion,
                criterion2 = test_criterion,
                device= device,
                val_random_seed = val_random_seed
            )
            if test_mpjpe <= best_test_mpjpe:
                print(f"best test mpjpe is:{test_mpjpe}")
                best_test_mpjpe = test_mpjpe
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion1, criterion2, device, val_random_seed):
    model.eval()
    VK_test_mpjpe = 0
    VK_test_pampjpe = 0
    VK_test_mse = 0

    depth_test_mpjpe = 0
    depth_test_pampjpe = 0
    depth_test_mse = 0

    lidar_test_mpjpe = 0
    lidar_test_pampjpe = 0
    lidar_test_mse = 0

    mmwave_test_mpjpe = 0
    mmwave_test_pampjpe = 0
    mmwave_test_mse = 0

    wifi_test_mpjpe = 0
    wifi_test_pampjpe = 0
    wifi_test_mse = 0

    VK_depth_test_mpjpe = 0
    VK_depth_test_pampjpe = 0
    VK_depth_test_mse = 0

    VK_lidar_test_mpjpe = 0
    VK_lidar_test_pampjpe = 0
    VK_lidar_test_mse = 0

    VK_mmwave_test_mpjpe = 0
    VK_mmwave_test_pampjpe = 0
    VK_mmwave_test_mse = 0

    VK_wifi_test_mpjpe = 0
    VK_wifi_test_pampjpe = 0
    VK_wifi_test_mse = 0

    depth_lidar_test_mpjpe = 0
    depth_lidar_test_pampjpe = 0
    depth_lidar_test_mse = 0

    depth_mmwave_test_mpjpe = 0
    depth_mmwave_test_pampjpe = 0
    depth_mmwave_test_mse = 0

    depth_wifi_test_mpjpe = 0
    depth_wifi_test_pampjpe = 0
    depth_wifi_test_mse = 0

    lidar_mmwave_test_mpjpe = 0
    lidar_mmwave_test_pampjpe = 0
    lidar_mmwave_test_mse = 0

    lidar_wifi_test_mpjpe = 0
    lidar_wifi_test_pampjpe = 0
    lidar_wifi_test_mse = 0

    mmwave_wifi_test_mpjpe = 0
    mmwave_wifi_test_pampjpe = 0
    mmwave_wifi_test_mse = 0

    VK_depth_lidar_test_mpjpe = 0
    VK_depth_lidar_test_pampjpe = 0
    VK_depth_lidar_test_mse = 0

    VK_depth_mmwave_test_mpjpe = 0
    VK_depth_mmwave_test_pampjpe = 0
    VK_depth_mmwave_test_mse = 0

    VK_depth_wifi_test_mpjpe = 0
    VK_depth_wifi_test_pampjpe = 0
    VK_depth_wifi_test_mse = 0

    VK_lidar_mmwave_test_mpjpe = 0
    VK_lidar_mmwave_test_pampjpe = 0
    VK_lidar_mmwave_test_mse = 0

    VK_lidar_wifi_test_mpjpe = 0
    VK_lidar_wifi_test_pampjpe = 0
    VK_lidar_wifi_test_mse = 0

    VK_mmwave_wifi_test_mpjpe = 0
    VK_mmwave_wifi_test_pampjpe = 0
    VK_mmwave_wifi_test_mse = 0

    depth_lidar_mmwave_test_mpjpe = 0
    depth_lidar_mmwave_test_pampjpe = 0
    depth_lidar_mmwave_test_mse = 0

    depth_lidar_wifi_test_mpjpe = 0
    depth_lidar_wifi_test_pampjpe = 0
    depth_lidar_wifi_test_mse = 0

    depth_mmwave_wifi_test_mpjpe = 0
    depth_mmwave_wifi_test_pampjpe = 0
    depth_mmwave_wifi_test_mse = 0

    lidar_mmwave_wifi_test_mpjpe = 0
    lidar_mmwave_wifi_test_pampjpe = 0
    lidar_mmwave_wifi_test_mse = 0

    VK_depth_lidar_mmwave_test_mpjpe = 0
    VK_depth_lidar_mmwave_test_pampjpe = 0
    VK_depth_lidar_mmwave_test_mse = 0

    VK_depth_lidar_wifi_test_mpjpe = 0
    VK_depth_lidar_wifi_test_pampjpe = 0
    VK_depth_lidar_wifi_test_mse = 0

    VK_depth_mmwave_wifi_test_mpjpe = 0
    VK_depth_mmwave_wifi_test_pampjpe = 0
    VK_depth_mmwave_wifi_test_mse = 0

    VK_lidar_mmwave_wifi_test_mpjpe = 0
    VK_lidar_mmwave_wifi_test_pampjpe = 0
    VK_lidar_mmwave_wifi_test_mse = 0

    depth_lidar_mmwave_wifi_test_mpjpe = 0
    depth_lidar_mmwave_wifi_test_pampjpe = 0
    depth_lidar_mmwave_wifi_test_mse = 0

    VK_depth_lidar_mmwave_wifi_test_mpjpe = 0
    VK_depth_lidar_mmwave_wifi_test_pampjpe = 0
    VK_depth_lidar_mmwave_wifi_test_mse = 0
    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        VK_data, depth_data, mmwave_data, lidar_data, wifi_data, kpts, _ = data
        VK_data = VK_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        wifi_data = wifi_data.to(device)
        kpts.to(device)
        labels = kpts.type(torch.FloatTensor)
        labels_ = labels.detach().numpy()


        ' SINGLE MODALITY '
        ### VK
        VK_modality_list = [True, False, False, False, False]
        VK_outputs = model(VK_data, depth_data, mmwave_data, lidar_data, wifi_data, VK_modality_list)
        VK_outputs = VK_outputs.type(torch.FloatTensor)
        VK_outputs.to(device)
        VK_test_mse += criterion1(VK_outputs,labels).item() * VK_data.size(0)
        VK_outputs = VK_outputs.detach().numpy()
        VK_mpjpe, VK_pampjpe = criterion2(VK_outputs,labels_)
        VK_test_mpjpe += VK_mpjpe.item() * VK_data.size(0)
        VK_test_pampjpe += VK_pampjpe.item() * VK_data.size(0)
        ### depth
        depth_modality_list = [False, True, False, False, False]
        depth_outputs = model(VK_data, depth_data, mmwave_data, lidar_data, wifi_data, depth_modality_list)
        depth_outputs = depth_outputs.type(torch.FloatTensor)
        depth_outputs.to(device)
        depth_test_mse += criterion1(depth_outputs,labels).item() * VK_data.size(0)
        depth_outputs = depth_outputs.detach().numpy()
        depth_mpjpe, depth_pampjpe = criterion2(depth_outputs,labels_)
        depth_test_mpjpe += depth_mpjpe.item() * VK_data.size(0)
        depth_test_pampjpe += depth_pampjpe.item() * VK_data.size(0)
        ### lidar
        lidar_modality_list = [False, False, False, True, False]
        lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_modality_list)
        lidar_outputs = lidar_outputs.type(torch.FloatTensor)
        lidar_outputs.to(device)
        lidar_test_mse += criterion1(lidar_outputs,labels).item() * VK_data.size(0)
        lidar_outputs = lidar_outputs.detach().numpy()
        lidar_mpjpe, lidar_pampjpe = criterion2(lidar_outputs,labels_)
        lidar_test_mpjpe += lidar_mpjpe.item() * VK_data.size(0)
        lidar_test_pampjpe += lidar_pampjpe.item() * VK_data.size(0)
        ### mmwave
        mmwave_modality_list = [False, False, True, False, False]
        mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, mmwave_modality_list)
        mmwave_outputs = mmwave_outputs.type(torch.FloatTensor)
        mmwave_outputs.to(device)
        mmwave_test_mse += criterion1(mmwave_outputs,labels).item() * VK_data.size(0)
        mmwave_outputs = mmwave_outputs.detach().numpy()
        mmwave_mpjpe, mmwave_pampjpe = criterion2(mmwave_outputs,labels_)
        mmwave_test_mpjpe += mmwave_mpjpe.item() * VK_data.size(0)
        mmwave_test_pampjpe += mmwave_pampjpe.item() * VK_data.size(0)
        ### wifi-cis
        wifi_modality_list = [False, False, False, False, True]
        wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, wifi_modality_list)
        wifi_outputs = wifi_outputs.type(torch.FloatTensor)
        wifi_outputs.to(device)
        wifi_test_mse += criterion1(wifi_outputs,labels).item() * VK_data.size(0)
        wifi_outputs = wifi_outputs.detach().numpy()
        wifi_mpjpe, wifi_pampjpe = criterion2(wifi_outputs,labels_)
        wifi_test_mpjpe += wifi_mpjpe.item() * VK_data.size(0)
        wifi_test_pampjpe += wifi_pampjpe.item() * VK_data.size(0)
        
        'Dual modality'
        ### VK + depth
        VK_depth_modality_list = [True, True, False, False, False]
        VK_depth_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_modality_list)
        VK_depth_outputs = VK_depth_outputs.type(torch.FloatTensor)
        VK_depth_outputs.to(device)
        VK_depth_test_mse += criterion1(VK_depth_outputs,labels).item() * VK_data.size(0)
        VK_depth_outputs = VK_depth_outputs.detach().numpy()
        VK_depth_mpjpe, VK_depth_pampjpe = criterion2(VK_depth_outputs,labels_)
        VK_depth_test_mpjpe += VK_depth_mpjpe.item() * VK_data.size(0)
        VK_depth_test_pampjpe += VK_depth_pampjpe.item() * VK_data.size(0)
        ### VK + lidar
        VK_lidar_modality_list = [True, False, False, True, False]
        VK_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_lidar_modality_list)
        VK_lidar_outputs = VK_lidar_outputs.type(torch.FloatTensor)
        VK_lidar_outputs.to(device)
        VK_lidar_test_mse += criterion1(VK_lidar_outputs,labels).item() * VK_data.size(0)
        VK_lidar_outputs = VK_lidar_outputs.detach().numpy()
        VK_lidar_mpjpe, VK_lidar_pampjpe = criterion2(VK_lidar_outputs,labels_)
        VK_lidar_test_mpjpe += VK_lidar_mpjpe.item() * VK_data.size(0)
        VK_lidar_test_pampjpe += VK_lidar_pampjpe.item() * VK_data.size(0)
        ### VK + mmwave
        VK_mmwave_modality_list = [True, False, True, False, False]
        VK_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_mmwave_modality_list)
        VK_mmwave_outputs = VK_mmwave_outputs.type(torch.FloatTensor)
        VK_mmwave_outputs.to(device)
        VK_mmwave_test_mse += criterion1(VK_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_mmwave_outputs = VK_mmwave_outputs.detach().numpy()
        VK_mmwave_mpjpe, VK_mmwave_pampjpe = criterion2(VK_mmwave_outputs,labels_)
        VK_mmwave_test_mpjpe += VK_mmwave_mpjpe.item() * VK_data.size(0)
        VK_mmwave_test_pampjpe += VK_mmwave_pampjpe.item() * VK_data.size(0)
        ### VK + wifi
        VK_wifi_modality_list = [True, False, False, False, True]
        VK_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_wifi_modality_list)
        VK_wifi_outputs = VK_wifi_outputs.type(torch.FloatTensor)
        VK_wifi_outputs.to(device)
        VK_wifi_test_mse += criterion1(VK_wifi_outputs,labels).item() * VK_data.size(0)
        VK_wifi_outputs = VK_wifi_outputs.detach().numpy()
        VK_wifi_mpjpe, VK_wifi_pampjpe = criterion2(VK_wifi_outputs,labels_)
        VK_wifi_test_mpjpe += VK_wifi_mpjpe.item() * VK_data.size(0)
        VK_wifi_test_pampjpe += VK_wifi_pampjpe.item() * VK_data.size(0)
        ### depth + lidar
        depth_lidar_modality_list = [False, True, False, True, False]
        depth_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_modality_list)
        depth_lidar_outputs = depth_lidar_outputs.type(torch.FloatTensor)
        depth_lidar_outputs.to(device)
        depth_lidar_test_mse += criterion1(depth_lidar_outputs,labels).item() * VK_data.size(0)
        depth_lidar_outputs = depth_lidar_outputs.detach().numpy()
        depth_lidar_mpjpe, depth_lidar_pampjpe = criterion2(depth_lidar_outputs,labels_)
        depth_lidar_test_mpjpe += depth_lidar_mpjpe.item() * VK_data.size(0)
        depth_lidar_test_pampjpe += depth_lidar_pampjpe.item() * VK_data.size(0)
        ### depth + mmwave
        depth_mmwave_modality_list = [False, True, True, False, False]
        depth_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_mmwave_modality_list)
        depth_mmwave_outputs = depth_mmwave_outputs.type(torch.FloatTensor)
        depth_mmwave_outputs.to(device)
        depth_mmwave_test_mse += criterion1(depth_mmwave_outputs,labels).item() * VK_data.size(0)
        depth_mmwave_outputs = depth_mmwave_outputs.detach().numpy()
        depth_mmwave_mpjpe, depth_mmwave_pampjpe = criterion2(depth_mmwave_outputs,labels_)
        depth_mmwave_test_mpjpe += depth_mmwave_mpjpe.item() * VK_data.size(0)
        depth_mmwave_test_pampjpe += depth_mmwave_pampjpe.item() * VK_data.size(0)
        ### depth + wifi
        depth_wifi_modality_list = [False, True, False, False, True]
        depth_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_wifi_modality_list)
        depth_wifi_outputs = depth_wifi_outputs.type(torch.FloatTensor)
        depth_wifi_outputs.to(device)
        depth_wifi_test_mse += criterion1(depth_wifi_outputs,labels).item() * VK_data.size(0)
        depth_wifi_outputs = depth_wifi_outputs.detach().numpy()
        depth_wifi_mpjpe, depth_wifi_pampjpe = criterion2(depth_wifi_outputs,labels_)
        depth_wifi_test_mpjpe += depth_wifi_mpjpe.item() * VK_data.size(0)
        depth_wifi_test_pampjpe += depth_wifi_pampjpe.item() * VK_data.size(0)
        ### lidar + mmwave
        lidar_mmwave_modality_list = [False, False, True, True, False]
        lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_mmwave_modality_list)
        lidar_mmwave_outputs = lidar_mmwave_outputs.type(torch.FloatTensor)
        lidar_mmwave_outputs.to(device)
        lidar_mmwave_test_mse += criterion1(lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        lidar_mmwave_outputs = lidar_mmwave_outputs.detach().numpy()
        lidar_mmwave_mpjpe, lidar_mmwave_pampjpe = criterion2(lidar_mmwave_outputs,labels_)
        lidar_mmwave_test_mpjpe += lidar_mmwave_mpjpe.item() * VK_data.size(0)
        lidar_mmwave_test_pampjpe += lidar_mmwave_pampjpe.item() * VK_data.size(0)
        ### lidar + wifi
        lidar_wifi_modality_list = [False, False, False, True, True]
        lidar_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_wifi_modality_list)
        lidar_wifi_outputs = lidar_wifi_outputs.type(torch.FloatTensor)
        lidar_wifi_outputs.to(device)
        lidar_wifi_test_mse += criterion1(lidar_wifi_outputs,labels).item() * VK_data.size(0)
        lidar_wifi_outputs = lidar_wifi_outputs.detach().numpy()
        lidar_wifi_mpjpe, lidar_wifi_pampjpe = criterion2(lidar_wifi_outputs,labels_)
        lidar_wifi_test_mpjpe += lidar_wifi_mpjpe.item() * VK_data.size(0)
        lidar_wifi_test_pampjpe += lidar_wifi_pampjpe.item() * VK_data.size(0)
        ### mmwave + wifi
        mmwave_wifi_modality_list = [False, False, True, False, True]
        mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, mmwave_wifi_modality_list)
        mmwave_wifi_outputs = mmwave_wifi_outputs.type(torch.FloatTensor)
        mmwave_wifi_outputs.to(device)
        mmwave_wifi_test_mse += criterion1(mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        mmwave_wifi_outputs = mmwave_wifi_outputs.detach().numpy()
        mmwave_wifi_mpjpe, mmwave_wifi_pampjpe = criterion2(mmwave_wifi_outputs,labels_)
        mmwave_wifi_test_mpjpe += mmwave_wifi_mpjpe.item() * VK_data.size(0)
        mmwave_wifi_test_pampjpe += mmwave_wifi_pampjpe.item() * VK_data.size(0)

        'Three modality'
        ### VK + depth + lidar
        VK_depth_lidar_modality_list = [True, True, False, True, False]
        VK_depth_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_lidar_modality_list)
        VK_depth_lidar_outputs = VK_depth_lidar_outputs.type(torch.FloatTensor)
        VK_depth_lidar_outputs.to(device)
        VK_depth_lidar_test_mse += criterion1(VK_depth_lidar_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_outputs = VK_depth_lidar_outputs.detach().numpy()
        VK_depth_lidar_mpjpe, VK_depth_lidar_pampjpe = criterion2(VK_depth_lidar_outputs,labels_)
        VK_depth_lidar_test_mpjpe += VK_depth_lidar_mpjpe.item() * VK_data.size(0)
        VK_depth_lidar_test_pampjpe += VK_depth_lidar_pampjpe.item() * VK_data.size(0)
        ### VK + depth + mmwave
        VK_depth_mmwave_modality_list = [True, True, True, False, False]
        VK_depth_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_mmwave_modality_list)
        VK_depth_mmwave_outputs = VK_depth_mmwave_outputs.type(torch.FloatTensor)
        VK_depth_mmwave_outputs.to(device)
        VK_depth_mmwave_test_mse += criterion1(VK_depth_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_depth_mmwave_outputs = VK_depth_mmwave_outputs.detach().numpy()
        VK_depth_mmwave_mpjpe, VK_depth_mmwave_pampjpe = criterion2(VK_depth_mmwave_outputs,labels_)
        VK_depth_mmwave_test_mpjpe += VK_depth_mmwave_mpjpe.item() * VK_data.size(0)
        VK_depth_mmwave_test_pampjpe += VK_depth_mmwave_pampjpe.item() * VK_data.size(0)
        ### VK + depth + wifi
        VK_depth_wifi_modality_list = [True, True, False, False, True]
        VK_depth_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_wifi_modality_list)
        VK_depth_wifi_outputs = VK_depth_wifi_outputs.type(torch.FloatTensor)
        VK_depth_wifi_outputs.to(device)
        VK_depth_wifi_test_mse += criterion1(VK_depth_wifi_outputs,labels).item() * VK_data.size(0)
        VK_depth_wifi_outputs = VK_depth_wifi_outputs.detach().numpy()
        VK_depth_wifi_mpjpe, VK_depth_wifi_pampjpe = criterion2(VK_depth_wifi_outputs,labels_)
        VK_depth_wifi_test_mpjpe += VK_depth_wifi_mpjpe.item() * VK_data.size(0)
        VK_depth_wifi_test_pampjpe += VK_depth_wifi_pampjpe.item() * VK_data.size(0)
        ### VK + lidar + mmwave
        VK_lidar_mmwave_modality_list = [True, False, True, True, False]
        VK_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_lidar_mmwave_modality_list)
        VK_lidar_mmwave_outputs = VK_lidar_mmwave_outputs.type(torch.FloatTensor)
        VK_lidar_mmwave_outputs.to(device)
        VK_lidar_mmwave_test_mse += criterion1(VK_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_lidar_mmwave_outputs = VK_lidar_mmwave_outputs.detach().numpy()
        VK_lidar_mmwave_mpjpe, VK_lidar_mmwave_pampjpe = criterion2(VK_lidar_mmwave_outputs,labels_)
        VK_lidar_mmwave_test_mpjpe += VK_lidar_mmwave_mpjpe.item() * VK_data.size(0)
        VK_lidar_mmwave_test_pampjpe += VK_lidar_mmwave_pampjpe.item() * VK_data.size(0)
        ### VK + lidar + wifi
        VK_lidar_wifi_modality_list = [True, False, False, True, True]
        VK_lidar_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_lidar_wifi_modality_list)
        VK_lidar_wifi_outputs = VK_lidar_wifi_outputs.type(torch.FloatTensor)
        VK_lidar_wifi_outputs.to(device)
        VK_lidar_wifi_test_mse += criterion1(VK_lidar_wifi_outputs,labels).item() * VK_data.size(0)
        VK_lidar_wifi_outputs = VK_lidar_wifi_outputs.detach().numpy()
        VK_lidar_wifi_mpjpe, VK_lidar_wifi_pampjpe = criterion2(VK_lidar_wifi_outputs,labels_)
        VK_lidar_wifi_test_mpjpe += VK_lidar_wifi_mpjpe.item() * VK_data.size(0)
        VK_lidar_wifi_test_pampjpe += VK_lidar_wifi_pampjpe.item() * VK_data.size(0)
        ### VK + mmwave + wifi
        VK_mmwave_wifi_modality_list = [True, False, True, False, True]
        VK_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_mmwave_wifi_modality_list)
        VK_mmwave_wifi_outputs = VK_mmwave_wifi_outputs.type(torch.FloatTensor)
        VK_mmwave_wifi_outputs.to(device)
        VK_mmwave_wifi_test_mse += criterion1(VK_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        VK_mmwave_wifi_outputs = VK_mmwave_wifi_outputs.detach().numpy()
        VK_mmwave_wifi_mpjpe, VK_mmwave_wifi_pampjpe = criterion2(VK_mmwave_wifi_outputs,labels_)
        VK_mmwave_wifi_test_mpjpe += VK_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        VK_mmwave_wifi_test_pampjpe += VK_mmwave_wifi_pampjpe.item() * VK_data.size(0)
        ### depth + lidar + mmwave
        depth_lidar_mmwave_modality_list = [False, True, True, True, False]
        depth_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_mmwave_modality_list)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_outputs.to(device)
        depth_lidar_mmwave_test_mse += criterion1(depth_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.detach().numpy()
        depth_lidar_mmwave_mpjpe, depth_lidar_mmwave_pampjpe = criterion2(depth_lidar_mmwave_outputs,labels_)
        depth_lidar_mmwave_test_mpjpe += depth_lidar_mmwave_mpjpe.item() * VK_data.size(0)
        depth_lidar_mmwave_test_pampjpe += depth_lidar_mmwave_pampjpe.item() * VK_data.size(0)
        ### depth + lidar + wifi
        depth_lidar_wifi_modality_list = [False, True, False, True, True]
        depth_lidar_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_wifi_modality_list)
        depth_lidar_wifi_outputs = depth_lidar_wifi_outputs.type(torch.FloatTensor)
        depth_lidar_wifi_outputs.to(device)
        depth_lidar_wifi_test_mse += criterion1(depth_lidar_wifi_outputs,labels).item() * VK_data.size(0)
        depth_lidar_wifi_outputs = depth_lidar_wifi_outputs.detach().numpy()
        depth_lidar_wifi_mpjpe, depth_lidar_wifi_pampjpe = criterion2(depth_lidar_wifi_outputs,labels_)
        depth_lidar_wifi_test_mpjpe += depth_lidar_wifi_mpjpe.item() * VK_data.size(0)
        depth_lidar_wifi_test_pampjpe += depth_lidar_wifi_pampjpe.item() * VK_data.size(0)
        ### depth + mmwave + wifi
        depth_mmwave_wifi_modality_list = [False, True, True, False, True]
        depth_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_mmwave_wifi_modality_list)
        depth_mmwave_wifi_outputs = depth_mmwave_wifi_outputs.type(torch.FloatTensor)
        depth_mmwave_wifi_outputs.to(device)
        depth_mmwave_wifi_test_mse += criterion1(depth_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        depth_mmwave_wifi_outputs = depth_mmwave_wifi_outputs.detach().numpy()
        depth_mmwave_wifi_mpjpe, depth_mmwave_wifi_pampjpe = criterion2(depth_mmwave_wifi_outputs,labels_)
        depth_mmwave_wifi_test_mpjpe += depth_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        depth_mmwave_wifi_test_pampjpe += depth_mmwave_wifi_pampjpe.item() * VK_data.size(0)
        ### lidar + mmwave + wifi
        lidar_mmwave_wifi_modality_list = [False, False, True, True, True]
        lidar_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, lidar_mmwave_wifi_modality_list)
        lidar_mmwave_wifi_outputs = lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        lidar_mmwave_wifi_outputs.to(device)
        lidar_mmwave_wifi_test_mse += criterion1(lidar_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        lidar_mmwave_wifi_outputs = lidar_mmwave_wifi_outputs.detach().numpy()
        lidar_mmwave_wifi_mpjpe, lidar_mmwave_wifi_pampjpe = criterion2(lidar_mmwave_wifi_outputs,labels_)
        lidar_mmwave_wifi_test_mpjpe += lidar_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        lidar_mmwave_wifi_test_pampjpe += lidar_mmwave_wifi_pampjpe.item() * VK_data.size(0)

        'Four modality'
        ### VK + depth + lidar + mmwave
        VK_depth_lidar_mmwave_modality_list = [True, True, True, True, False]
        VK_depth_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_lidar_mmwave_modality_list)
        VK_depth_lidar_mmwave_outputs = VK_depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        VK_depth_lidar_mmwave_outputs.to(device)
        VK_depth_lidar_mmwave_test_mse += criterion1(VK_depth_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_mmwave_outputs = VK_depth_lidar_mmwave_outputs.detach().numpy()
        VK_depth_lidar_mmwave_mpjpe, VK_depth_lidar_mmwave_pampjpe = criterion2(VK_depth_lidar_mmwave_outputs,labels_)
        VK_depth_lidar_mmwave_test_mpjpe += VK_depth_lidar_mmwave_mpjpe.item() * VK_data.size(0)
        VK_depth_lidar_mmwave_test_pampjpe += VK_depth_lidar_mmwave_pampjpe.item() * VK_data.size(0)
        ### VK + depth + lidar + wifi
        VK_depth_lidar_wifi_modality_list = [True, True, False, True, True]
        VK_depth_lidar_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_lidar_wifi_modality_list)
        VK_depth_lidar_wifi_outputs = VK_depth_lidar_wifi_outputs.type(torch.FloatTensor)
        VK_depth_lidar_wifi_outputs.to(device)
        VK_depth_lidar_wifi_test_mse += criterion1(VK_depth_lidar_wifi_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_wifi_outputs = VK_depth_lidar_wifi_outputs.detach().numpy()
        VK_depth_lidar_wifi_mpjpe, VK_depth_lidar_wifi_pampjpe = criterion2(VK_depth_lidar_wifi_outputs,labels_)
        VK_depth_lidar_wifi_test_mpjpe += VK_depth_lidar_wifi_mpjpe.item() * VK_data.size(0)
        VK_depth_lidar_wifi_test_pampjpe += VK_depth_lidar_wifi_pampjpe.item() * VK_data.size(0)
        ### VK + depth + mmwave + wifi
        VK_depth_mmwave_wifi_modality_list = [True, True, True, False, True]
        VK_depth_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_mmwave_wifi_modality_list)
        VK_depth_mmwave_wifi_outputs = VK_depth_mmwave_wifi_outputs.type(torch.FloatTensor)
        VK_depth_mmwave_wifi_outputs.to(device)
        VK_depth_mmwave_wifi_test_mse += criterion1(VK_depth_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        VK_depth_mmwave_wifi_outputs = VK_depth_mmwave_wifi_outputs.detach().numpy()
        VK_depth_mmwave_wifi_mpjpe, VK_depth_mmwave_wifi_pampjpe = criterion2(VK_depth_mmwave_wifi_outputs,labels_)
        VK_depth_mmwave_wifi_test_mpjpe += VK_depth_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        VK_depth_mmwave_wifi_test_pampjpe += VK_depth_mmwave_wifi_pampjpe.item() * VK_data.size(0)
        ### VK + lidar + mmwave + wifi
        VK_lidar_mmwave_wifi_modality_list = [True, False, True, True, True]
        VK_lidar_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_lidar_mmwave_wifi_modality_list)
        VK_lidar_mmwave_wifi_outputs = VK_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        VK_lidar_mmwave_wifi_outputs.to(device)
        VK_lidar_mmwave_wifi_test_mse += criterion1(VK_lidar_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        VK_lidar_mmwave_wifi_outputs = VK_lidar_mmwave_wifi_outputs.detach().numpy()
        VK_lidar_mmwave_wifi_mpjpe, VK_lidar_mmwave_wifi_pampjpe = criterion2(VK_lidar_mmwave_wifi_outputs,labels_)
        VK_lidar_mmwave_wifi_test_mpjpe += VK_lidar_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        VK_lidar_mmwave_wifi_test_pampjpe += VK_lidar_mmwave_wifi_pampjpe.item() * VK_data.size(0)
        ### depth + lidar + mmwave + wifi
        depth_lidar_mmwave_wifi_modality_list = [False, True, True, True, True]
        depth_lidar_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, depth_lidar_mmwave_wifi_modality_list)
        depth_lidar_mmwave_wifi_outputs = depth_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_wifi_outputs.to(device)
        depth_lidar_mmwave_wifi_test_mse += criterion1(depth_lidar_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        depth_lidar_mmwave_wifi_outputs = depth_lidar_mmwave_wifi_outputs.detach().numpy()
        depth_lidar_mmwave_wifi_mpjpe, depth_lidar_mmwave_wifi_pampjpe = criterion2(depth_lidar_mmwave_wifi_outputs,labels_)
        depth_lidar_mmwave_wifi_test_mpjpe += depth_lidar_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        depth_lidar_mmwave_wifi_test_pampjpe += depth_lidar_mmwave_wifi_pampjpe.item() * VK_data.size(0)

        'ALL modality'
        ### VK + depth + lidar + mmwave + wifi
        VK_depth_lidar_mmwave_wifi_modality_list = [True, True, True, True, True]
        VK_depth_lidar_mmwave_wifi_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, wifi_data, VK_depth_lidar_mmwave_wifi_modality_list)
        VK_depth_lidar_mmwave_wifi_outputs = VK_depth_lidar_mmwave_wifi_outputs.type(torch.FloatTensor)
        VK_depth_lidar_mmwave_wifi_outputs.to(device)
        VK_depth_lidar_mmwave_wifi_test_mse += criterion1(VK_depth_lidar_mmwave_wifi_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_mmwave_wifi_outputs = VK_depth_lidar_mmwave_wifi_outputs.detach().numpy()
        VK_depth_lidar_mmwave_wifi_mpjpe, VK_depth_lidar_mmwave_wifi_pampjpe = criterion2(VK_depth_lidar_mmwave_wifi_outputs,labels_)
        VK_depth_lidar_mmwave_wifi_test_mpjpe += VK_depth_lidar_mmwave_wifi_mpjpe.item() * VK_data.size(0)
        VK_depth_lidar_mmwave_wifi_test_pampjpe += VK_depth_lidar_mmwave_wifi_pampjpe.item() * VK_data.size(0)


    'single modality'
    ### VK
    VK_test_mpjpe = VK_test_mpjpe/len(tensor_loader.dataset)
    VK_test_pampjpe = VK_test_pampjpe/len(tensor_loader.dataset)
    VK_test_mse = VK_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK',float(VK_test_mse), float(VK_test_mpjpe),float(VK_test_pampjpe)))
    ### depth
    depth_test_mpjpe = depth_test_mpjpe/len(tensor_loader.dataset)
    depth_test_pampjpe = depth_test_pampjpe/len(tensor_loader.dataset)
    depth_test_mse = depth_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth',float(depth_test_mse), float(depth_test_mpjpe),float(depth_test_pampjpe)))
    ### lidar
    lidar_test_mpjpe = lidar_test_mpjpe/len(tensor_loader.dataset)
    lidar_test_pampjpe = lidar_test_pampjpe/len(tensor_loader.dataset)
    lidar_test_mse = lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar',float(lidar_test_mse), float(lidar_test_mpjpe),float(lidar_test_pampjpe)))
    ### mmwave
    mmwave_test_mpjpe = mmwave_test_mpjpe/len(tensor_loader.dataset)
    mmwave_test_pampjpe = mmwave_test_pampjpe/len(tensor_loader.dataset)
    mmwave_test_mse = mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('mmWave',float(mmwave_test_mse), float(mmwave_test_mpjpe),float(mmwave_test_pampjpe)))
    ### wifi
    wifi_test_mpjpe = wifi_test_mpjpe/len(tensor_loader.dataset)
    wifi_test_pampjpe = wifi_test_pampjpe/len(tensor_loader.dataset)
    wifi_test_mse = wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('WiFi-CSI',float(wifi_test_mse), float(wifi_test_mpjpe),float(wifi_test_pampjpe)))
    
    'dual modality'
    ### VK + depth
    VK_depth_test_mpjpe = VK_depth_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_test_pampjpe = VK_depth_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_test_mse = VK_depth_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+Depth',float(VK_depth_test_mse), float(VK_depth_test_mpjpe),float(VK_depth_test_pampjpe)))
    ### VK + lidar
    VK_lidar_test_mpjpe = VK_lidar_test_mpjpe/len(tensor_loader.dataset)
    VK_lidar_test_pampjpe = VK_lidar_test_pampjpe/len(tensor_loader.dataset)
    VK_lidar_test_mse = VK_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+Lidar',float(VK_lidar_test_mse), float(VK_lidar_test_mpjpe),float(VK_lidar_test_pampjpe)))
    ### VK + mmwave
    VK_mmwave_test_mpjpe = VK_mmwave_test_mpjpe/len(tensor_loader.dataset)
    VK_mmwave_test_pampjpe = VK_mmwave_test_pampjpe/len(tensor_loader.dataset)
    VK_mmwave_test_mse = VK_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+mmWave',float(VK_mmwave_test_mse), float(VK_mmwave_test_mpjpe),float(VK_mmwave_test_pampjpe)))
    ### VK + wifi
    VK_wifi_test_mpjpe = VK_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_wifi_test_pampjpe = VK_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_wifi_test_mse = VK_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+WiFi-CSI',float(VK_wifi_test_mse), float(VK_wifi_test_mpjpe),float(VK_wifi_test_pampjpe)))
    ### depth + lidar
    depth_lidar_test_mpjpe = depth_lidar_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_test_pampjpe = depth_lidar_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_test_mse = depth_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar',float(depth_lidar_test_mse), float(depth_lidar_test_mpjpe),float(depth_lidar_test_pampjpe)))
    ### depth + mmwave
    depth_mmwave_test_mpjpe = depth_mmwave_test_mpjpe/len(tensor_loader.dataset)
    depth_mmwave_test_pampjpe = depth_mmwave_test_pampjpe/len(tensor_loader.dataset)
    depth_mmwave_test_mse = depth_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+mmWave',float(depth_mmwave_test_mse), float(depth_mmwave_test_mpjpe),float(depth_mmwave_test_pampjpe)))
    ### depth + wifi
    depth_wifi_test_mpjpe = depth_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_wifi_test_pampjpe = depth_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_wifi_test_mse = depth_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+WiFi-CSI',float(depth_wifi_test_mse), float(depth_wifi_test_mpjpe),float(depth_wifi_test_pampjpe)))
    ### lidar + mmwave
    lidar_mmwave_test_mpjpe = lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    lidar_mmwave_test_pampjpe = lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    lidar_mmwave_test_mse = lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar+mmWave',float(lidar_mmwave_test_mse), float(lidar_mmwave_test_mpjpe),float(lidar_mmwave_test_pampjpe)))
    ### lidar + wifi
    lidar_wifi_test_mpjpe = lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    lidar_wifi_test_pampjpe = lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    lidar_wifi_test_mse = lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Lidar+WiFi-CSI',float(lidar_wifi_test_mse), float(lidar_wifi_test_mpjpe),float(lidar_wifi_test_pampjpe)))
    ### mmwave + wifi
    mmwave_wifi_test_mpjpe = mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    mmwave_wifi_test_pampjpe = mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    mmwave_wifi_test_mse = mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('mmWave+WiFi-CSI',float(mmwave_wifi_test_mse), float(mmwave_wifi_test_mpjpe),float(mmwave_wifi_test_pampjpe)))
    
    'three modality'
    ### VK + depth + lidar
    VK_depth_lidar_test_mpjpe = VK_depth_lidar_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_lidar_test_pampjpe = VK_depth_lidar_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_lidar_test_mse = VK_depth_lidar_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Depth+Lidar',float(VK_depth_lidar_test_mse), float(VK_depth_lidar_test_mpjpe),float(VK_depth_lidar_test_pampjpe)))
    ### VK + depth + mmwave
    VK_depth_mmwave_test_mpjpe = VK_depth_mmwave_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_mmwave_test_pampjpe = VK_depth_mmwave_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_mmwave_test_mse = VK_depth_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Depth+mmWave',float(VK_depth_mmwave_test_mse), float(VK_depth_mmwave_test_mpjpe),float(VK_depth_mmwave_test_pampjpe)))
    ### VK + depth + wifi
    VK_depth_wifi_test_mpjpe = VK_depth_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_wifi_test_pampjpe = VK_depth_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_wifi_test_mse = VK_depth_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Depth+WiFi-CSI',float(VK_depth_wifi_test_mse), float(VK_depth_wifi_test_mpjpe),float(VK_depth_wifi_test_pampjpe)))
    ### VK + lidar + mmwave
    VK_lidar_mmwave_test_mpjpe = VK_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    VK_lidar_mmwave_test_pampjpe = VK_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    VK_lidar_mmwave_test_mse = VK_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Lidar+mmWave',float(VK_lidar_mmwave_test_mse), float(VK_lidar_mmwave_test_mpjpe),float(VK_lidar_mmwave_test_pampjpe)))
    ### VK + lidar + wifi
    VK_lidar_wifi_test_mpjpe = VK_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_lidar_wifi_test_pampjpe = VK_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_lidar_wifi_test_mse = VK_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Lidar+WiFi-CSI',float(VK_lidar_wifi_test_mse), float(VK_lidar_wifi_test_mpjpe),float(VK_lidar_wifi_test_pampjpe)))
    ### VK + mmwave + wifi
    VK_mmwave_wifi_test_mpjpe = VK_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_mmwave_wifi_test_pampjpe = VK_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_mmwave_wifi_test_mse = VK_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+mmWave+WiFi-CSI',float(VK_mmwave_wifi_test_mse), float(VK_mmwave_wifi_test_mpjpe),float(VK_mmwave_wifi_test_pampjpe)))
    ### depth + lidar + mmwave
    depth_lidar_mmwave_test_mpjpe = depth_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_pampjpe = depth_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_mse = depth_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar+mmWave',float(depth_lidar_mmwave_test_mse), float(depth_lidar_mmwave_test_mpjpe),float(depth_lidar_mmwave_test_pampjpe)))
    ### depth + lidar + wifi
    depth_lidar_wifi_test_mpjpe = depth_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_wifi_test_pampjpe = depth_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_wifi_test_mse = depth_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('Depth+Lidar+WiFi-CSI',float(depth_lidar_wifi_test_mse), float(depth_lidar_wifi_test_mpjpe),float(depth_lidar_wifi_test_pampjpe)))
    ### depth + mmwave + wifi
    depth_mmwave_wifi_test_mpjpe = depth_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_mmwave_wifi_test_pampjpe = depth_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_mmwave_wifi_test_mse = depth_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Depth+mmWave+WiFi-CSI',float(depth_mmwave_wifi_test_mse), float(depth_mmwave_wifi_test_mpjpe),float(depth_mmwave_wifi_test_pampjpe)))
    ### lidar + mmwave + wifi
    lidar_mmwave_wifi_test_mpjpe = lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    lidar_mmwave_wifi_test_pampjpe = lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    lidar_mmwave_wifi_test_mse = lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Lidar+mmWave+WiFi-CSI',float(lidar_mmwave_wifi_test_mse), float(lidar_mmwave_wifi_test_mpjpe),float(lidar_mmwave_wifi_test_pampjpe)))
    
    'four modality'
    ### VK + depth + lidar + mmwave
    VK_depth_lidar_mmwave_test_mpjpe = VK_depth_lidar_mmwave_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_lidar_mmwave_test_pampjpe = VK_depth_lidar_mmwave_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_lidar_mmwave_test_mse = VK_depth_lidar_mmwave_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Depth+Lidar+mmWave',float(VK_depth_lidar_mmwave_test_mse), float(VK_depth_lidar_mmwave_test_mpjpe),float(VK_depth_lidar_mmwave_test_pampjpe)))
    ### VK + depth + lidar + wifi
    VK_depth_lidar_wifi_test_mpjpe = VK_depth_lidar_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_lidar_wifi_test_pampjpe = VK_depth_lidar_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_lidar_wifi_test_mse = VK_depth_lidar_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('VK+Depth+Lidar+WiFi-CSI',float(VK_depth_lidar_wifi_test_mse), float(VK_depth_lidar_wifi_test_mpjpe),float(VK_depth_lidar_wifi_test_pampjpe)))
    ### VK + depth + mmwave + wifi
    VK_depth_mmwave_wifi_test_mpjpe = VK_depth_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_mmwave_wifi_test_pampjpe = VK_depth_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_mmwave_wifi_test_mse = VK_depth_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+Depth+mmWave+WiFi-CSI',float(VK_depth_mmwave_wifi_test_mse), float(VK_depth_mmwave_wifi_test_mpjpe),float(VK_depth_mmwave_wifi_test_pampjpe)))
    ### VK + lidar + mmwave + wifi
    VK_lidar_mmwave_wifi_test_mpjpe = VK_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_lidar_mmwave_wifi_test_pampjpe = VK_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_lidar_mmwave_wifi_test_mse = VK_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+Lidar+mmWave+WiFi-CSI',float(VK_lidar_mmwave_wifi_test_mse), float(VK_lidar_mmwave_wifi_test_mpjpe),float(VK_lidar_mmwave_wifi_test_pampjpe)))
    ### depth + lidar + mmwave + wifi
    depth_lidar_mmwave_wifi_test_mpjpe = depth_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_wifi_test_pampjpe = depth_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    depth_lidar_mmwave_wifi_test_mse = depth_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}\n".format('Depth+Lidar+mmWave+WiFi-CSI',float(depth_lidar_mmwave_wifi_test_mse), float(depth_lidar_mmwave_wifi_test_mpjpe),float(depth_lidar_mmwave_wifi_test_pampjpe)))

    'ALL modality'
    ### VK + depth + lidar + mmwave + wifi
    VK_depth_lidar_mmwave_wifi_test_mpjpe = VK_depth_lidar_mmwave_wifi_test_mpjpe/len(tensor_loader.dataset)
    VK_depth_lidar_mmwave_wifi_test_pampjpe = VK_depth_lidar_mmwave_wifi_test_pampjpe/len(tensor_loader.dataset)
    VK_depth_lidar_mmwave_wifi_test_mse = VK_depth_lidar_mmwave_wifi_test_mse/len(tensor_loader.dataset)
    print("modality: {}, mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format('VK+Depth+Lidar+mmWave+WiFi-CSI',float(VK_depth_lidar_mmwave_wifi_test_mse), float(VK_depth_lidar_mmwave_wifi_test_mpjpe),float(VK_depth_lidar_mmwave_wifi_test_pampjpe)))
    return

