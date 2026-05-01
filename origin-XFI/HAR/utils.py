import numpy as np
import torch
import random
from tqdm import tqdm
from datetime import datetime

def generate_none_empth_modality_list():
    VK_ = random.choices(
        [True, False],
        k= 1,
        weights=[50, 50]
    )
    depth_ = random.choices(
        [True, False],
        k= 1,
        weights=[60, 40]
    )
    mmwave_ = random.choices(
        [True, False],
        k= 1,
        weights=[50, 50]
    )
    lidar_ = random.choices(
        [True, False],
        k= 1,
        weights=[90, 10]
    )
    modality_list = VK_ + depth_ + mmwave_ + lidar_
    if sum(modality_list) == 0:
        modality_list = generate_none_empth_modality_list()
        return modality_list
    else:
        return modality_list

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    dict_keys(['modality', 'scene', 'subject', 'action', 'idx', 'output',
    'input_vk', 'input_depth', 'input_lidar', 'input_mmwave'])
    '''
    all_actions = {'A01': 0., 'A02': 1., 'A03': 2., 'A04': 3., 'A05': 4., 
                   'A06': 5., 'A07': 6., 'A08': 7., 'A09': 8., 'A10': 9.,
                   'A11': 10., 'A12': 11., 'A13': 12., 'A14': 13., 'A15': 14., 
                   'A16': 15., 'A17': 16., 'A18': 17., 'A19': 18., 'A20': 19., 
                   'A21': 20., 'A22': 21., 'A23': 22., 'A24': 23., 'A25': 24., 
                   'A26': 25., 'A27': 26.}
    labels = []
    [labels.append(all_actions[t['action']]) for t in batch]
    labels = torch.FloatTensor(labels)

    lengths = torch.tensor([t['input_mmwave'].shape[0] for t in batch ])

    # vk
    VK_data = np.array([(t['input_vk']) for t in batch ])
    VK_data = torch.FloatTensor(VK_data)

    # depth
    depth_data = np.array([(t['input_depth']) for t in batch ])
    depth_data = torch.FloatTensor(depth_data).permute(0,3,1,2)

    # mmwave
    mmwave_data = [torch.Tensor(t['input_mmwave']) for t in batch ]
    mmwave_data = torch.nn.utils.rnn.pad_sequence(mmwave_data)
    mmwave_data = mmwave_data.permute(1,0,2)

    # lidar
    lidar_data = [torch.Tensor(t['input_lidar'].copy()) for t in batch ]
    lidar_data = torch.nn.utils.rnn.pad_sequence(lidar_data)
    lidar_data = lidar_data.permute(1,0,2)
    
    modality_list = generate_none_empth_modality_list()

    return VK_data, depth_data, mmwave_data, lidar_data, labels, modality_list

def har_test(model, tensor_loader, criterion, device,val_random_seed):
    model.eval()
    test_acc = 0
    test_loss = 0
    random.seed(val_random_seed)
    for i, data in enumerate(tqdm(tensor_loader)):
        VK_data, depth_data, mmwave_data, lidar_data, label, modality_list = data
        VK_data = VK_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        label.to(device)
        labels = label.type(torch.LongTensor)
        outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, modality_list)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item()/labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * labels.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return test_acc

def har_train(model, train_loader, test_loader, num_epochs, learning_rate, criterion, device, save_dir, val_random_seed):
    optimizer = torch.optim.AdamW(
        [
                {'params': model.feature_extractor.vk_extractor.parameters()},
                {'params': model.linear_projector.parameters()},
                {'params': model.X_Fusion_block.parameters()}
            ],
        lr = learning_rate
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15],gamma=0.1)
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parameter_dir = save_dir + '/checkpoint_' + now_time + '.pth'
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        random.seed(epoch)
        num_iter = 1000
        for i, data in enumerate(tqdm(train_loader)):
            if i < num_iter:
                VK_data, depth_data, mmwave_data, lidar_data, label, modality_list = data
                VK_data = VK_data.to(device)
                depth_data = depth_data.to(device)
                lidar_data = lidar_data.to(device)
                mmwave_data = mmwave_data.to(device)
                labels = label.to(device)
                labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = model(VK_data, depth_data, mmwave_data, lidar_data, modality_list)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                loss = criterion(outputs,labels)
                if loss == float('nan'):
                    print('nan')
                    print(outputs)
                    print(labels)
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                predict_y = torch.argmax(outputs,dim=1).to(device)
                epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
            else:
                break
        epoch_loss = epoch_loss/num_iter
        epoch_accuracy = epoch_accuracy/num_iter
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
        if (epoch+1) % 10 == 0:
            test_acc = har_test(
                model=model,
                tensor_loader=test_loader,
                criterion = criterion,
                device= device,
                val_random_seed=val_random_seed
            )
            if test_acc >= best_test_acc:
                print(f"best test accuracy is:{test_acc}")
                best_test_acc = test_acc
        scheduler.step()
    torch.save(model.state_dict(), parameter_dir)
    return

def multi_test(model, tensor_loader, criterion, device, val_random_seed):
    model.eval()
    VK_test_loss = 0
    VK_test_accuracy = 0

    depth_test_loss = 0
    depth_test_accuracy = 0
    

    lidar_test_loss = 0
    lidar_test_accuracy = 0

    mmwave_test_loss = 0
    mmwave_test_accuracy = 0

    VK_depth_test_loss = 0
    VK_depth_test_accuracy = 0

    VK_lidar_test_loss = 0
    VK_lidar_test_accuracy = 0

    VK_mmwave_test_loss = 0
    VK_mmwave_test_accuracy = 0

    depth_lidar_test_loss = 0
    depth_lidar_test_accuracy = 0

    depth_mmwave_test_loss = 0
    depth_mmwave_test_accuracy = 0

    lidar_mmwave_test_loss = 0
    lidar_mmwave_test_accuracy = 0

    VK_depth_lidar_test_loss = 0
    VK_depth_lidar_test_accuracy = 0

    VK_depth_mmwave_test_loss = 0
    VK_depth_mmwave_test_accuracy = 0

    VK_lidar_mmwave_test_loss = 0
    VK_lidar_mmwave_test_accuracy = 0

    depth_lidar_mmwave_test_loss = 0
    depth_lidar_mmwave_test_accuracy = 0

    VK_depth_lidar_mmwave_test_loss = 0
    VK_depth_lidar_mmwave_test_accuracy = 0

    random.seed(val_random_seed)
    for data in tqdm(tensor_loader):
        VK_data, depth_data, mmwave_data, lidar_data, label, _ = data
        VK_data = VK_data.to(device)
        depth_data = depth_data.to(device)
        lidar_data = lidar_data.to(device)
        mmwave_data = mmwave_data.to(device)
        label.to(device)
        labels = label.type(torch.LongTensor)


        ' SINGLE MODALITY '
        ### VK
        VK_modality_list = [True, False, False, False]
        VK_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_modality_list)
        VK_outputs = VK_outputs.type(torch.FloatTensor)
        VK_outputs.to(device)
        VK_test_loss += criterion(VK_outputs,labels).item() * VK_data.size(0)
        VK_predict_y = torch.argmax(VK_outputs,dim=1).to(device)
        VK_test_accuracy += (VK_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_outputs = VK_outputs.detach().cpu()
        VK_predict_y = VK_predict_y.detach().cpu()
        ### depth
        depth_modality_list = [False, True, False, False]
        depth_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, depth_modality_list)
        depth_outputs = depth_outputs.type(torch.FloatTensor)
        depth_outputs.to(device)
        depth_test_loss += criterion(depth_outputs,labels).item() * VK_data.size(0)
        depth_predict_y = torch.argmax(depth_outputs,dim=1).to(device)
        depth_test_accuracy += (depth_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_outputs = depth_outputs.detach().cpu()
        depth_predict_y = depth_predict_y.detach().cpu()
        ### lidar
        lidar_modality_list = [False, False, False, True]
        lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, lidar_modality_list)
        lidar_outputs = lidar_outputs.type(torch.FloatTensor)
        lidar_outputs.to(device)
        lidar_test_loss += criterion(lidar_outputs,labels).item() * VK_data.size(0)
        lidar_predict_y = torch.argmax(lidar_outputs,dim=1).to(device)
        lidar_test_accuracy += (lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        lidar_outputs = lidar_outputs.detach().cpu()
        lidar_predict_y = lidar_predict_y.detach().cpu()
        ### mmwave
        mmwave_modality_list = [False, False, True, False]
        mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, mmwave_modality_list)
        mmwave_outputs = mmwave_outputs.type(torch.FloatTensor)
        mmwave_outputs.to(device)
        mmwave_test_loss += criterion(mmwave_outputs,labels).item() * VK_data.size(0)
        mmwave_predict_y = torch.argmax(mmwave_outputs,dim=1).to(device)
        mmwave_test_accuracy += (mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        mmwave_outputs = mmwave_outputs.detach().cpu()
        mmwave_predict_y = mmwave_predict_y.detach().cpu()
        
        'Dual modality'
        ### VK + depth
        VK_depth_modality_list = [True, True, False, False]
        VK_depth_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_depth_modality_list)
        VK_depth_outputs = VK_depth_outputs.type(torch.FloatTensor)
        VK_depth_outputs.to(device)
        VK_depth_test_loss += criterion(VK_depth_outputs,labels).item() * VK_data.size(0)
        VK_depth_predict_y = torch.argmax(VK_depth_outputs,dim=1).to(device)
        VK_depth_test_accuracy += (VK_depth_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_depth_outputs = VK_depth_outputs.detach().cpu()
        VK_depth_predict_y = VK_depth_predict_y.detach().cpu()
        ### VK + lidar
        VK_lidar_modality_list = [True, False, False, True]
        VK_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_lidar_modality_list)
        VK_lidar_outputs = VK_lidar_outputs.type(torch.FloatTensor)
        VK_lidar_outputs.to(device)
        VK_lidar_test_loss += criterion(VK_lidar_outputs,labels).item() * VK_data.size(0)
        VK_lidar_predict_y = torch.argmax(VK_lidar_outputs,dim=1).to(device)
        VK_lidar_test_accuracy += (VK_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_lidar_outputs = VK_lidar_outputs.detach().cpu()
        VK_lidar_predict_y = VK_lidar_predict_y.detach().cpu()
        ### VK + mmwave
        VK_mmwave_modality_list = [True, False, True, False]
        VK_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_mmwave_modality_list)
        VK_mmwave_outputs = VK_mmwave_outputs.type(torch.FloatTensor)
        VK_mmwave_outputs.to(device)
        VK_mmwave_test_loss += criterion(VK_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_mmwave_predict_y = torch.argmax(VK_mmwave_outputs,dim=1).to(device)
        VK_mmwave_test_accuracy += (VK_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_mmwave_outputs = VK_mmwave_outputs.detach().cpu()
        VK_mmwave_predict_y = VK_mmwave_predict_y.detach().cpu()
        ### depth + lidar
        depth_lidar_modality_list = [False, True, False, True]
        depth_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, depth_lidar_modality_list)
        depth_lidar_outputs = depth_lidar_outputs.type(torch.FloatTensor)
        depth_lidar_outputs.to(device)
        depth_lidar_test_loss += criterion(depth_lidar_outputs,labels).item() * VK_data.size(0)
        depth_lidar_predict_y = torch.argmax(depth_lidar_outputs,dim=1).to(device)
        depth_lidar_test_accuracy += (depth_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_lidar_outputs = depth_lidar_outputs.detach().cpu()
        depth_lidar_predict_y = depth_lidar_predict_y.detach().cpu()
        ### depth + mmwave
        depth_mmwave_modality_list = [False, True, True, False]
        depth_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, depth_mmwave_modality_list)
        depth_mmwave_outputs = depth_mmwave_outputs.type(torch.FloatTensor)
        depth_mmwave_outputs.to(device)
        depth_mmwave_test_loss += criterion(depth_mmwave_outputs,labels).item() * VK_data.size(0)
        depth_mmwave_predict_y = torch.argmax(depth_mmwave_outputs,dim=1).to(device)
        depth_mmwave_test_accuracy += (depth_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_mmwave_outputs = depth_mmwave_outputs.detach().cpu()
        depth_mmwave_predict_y = depth_mmwave_predict_y.detach().cpu()
        ### lidar + mmwave
        lidar_mmwave_modality_list = [False, False, True, True]
        lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, lidar_mmwave_modality_list)
        lidar_mmwave_outputs = lidar_mmwave_outputs.type(torch.FloatTensor)
        lidar_mmwave_outputs.to(device)
        lidar_mmwave_test_loss += criterion(lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        lidar_mmwave_predict_y = torch.argmax(lidar_mmwave_outputs,dim=1).to(device)
        lidar_mmwave_test_accuracy += (lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        lidar_mmwave_outputs = lidar_mmwave_outputs.detach().cpu()
        lidar_mmwave_predict_y = lidar_mmwave_predict_y.detach().cpu()

        'Three modality'
        ### VK + depth + lidar
        VK_depth_lidar_modality_list = [True, True, False, True]
        VK_depth_lidar_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_depth_lidar_modality_list)
        VK_depth_lidar_outputs = VK_depth_lidar_outputs.type(torch.FloatTensor)
        VK_depth_lidar_outputs.to(device)
        VK_depth_lidar_test_loss += criterion(VK_depth_lidar_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_predict_y = torch.argmax(VK_depth_lidar_outputs,dim=1).to(device)
        VK_depth_lidar_test_accuracy += (VK_depth_lidar_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_depth_lidar_outputs = VK_depth_lidar_outputs.detach().cpu()
        VK_depth_lidar_predict_y = VK_depth_lidar_predict_y.detach().cpu()
        ### VK + depth + mmwave
        VK_depth_mmwave_modality_list = [True, True, True, False]
        VK_depth_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_depth_mmwave_modality_list)
        VK_depth_mmwave_outputs = VK_depth_mmwave_outputs.type(torch.FloatTensor)
        VK_depth_mmwave_outputs.to(device)
        VK_depth_mmwave_test_loss += criterion(VK_depth_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_depth_mmwave_predict_y = torch.argmax(VK_depth_mmwave_outputs,dim=1).to(device)
        VK_depth_mmwave_test_accuracy += (VK_depth_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_depth_mmwave_outputs = VK_depth_mmwave_outputs.detach().cpu()
        VK_depth_mmwave_predict_y = VK_depth_mmwave_predict_y.detach().cpu()
        ### VK + lidar + mmwave
        VK_lidar_mmwave_modality_list = [True, False, True, True]
        VK_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_lidar_mmwave_modality_list)
        VK_lidar_mmwave_outputs = VK_lidar_mmwave_outputs.type(torch.FloatTensor)
        VK_lidar_mmwave_outputs.to(device)
        VK_lidar_mmwave_test_loss += criterion(VK_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_lidar_mmwave_predict_y = torch.argmax(VK_lidar_mmwave_outputs,dim=1).to(device)
        VK_lidar_mmwave_test_accuracy += (VK_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_lidar_mmwave_outputs = VK_lidar_mmwave_outputs.detach().cpu()
        VK_lidar_mmwave_predict_y = VK_lidar_mmwave_predict_y.detach().cpu()
        ### depth + lidar + mmwave
        depth_lidar_mmwave_modality_list = [False, True, True, True]
        depth_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, depth_lidar_mmwave_modality_list)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        depth_lidar_mmwave_outputs.to(device)
        depth_lidar_mmwave_test_loss += criterion(depth_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        depth_lidar_mmwave_predict_y = torch.argmax(depth_lidar_mmwave_outputs,dim=1).to(device)
        depth_lidar_mmwave_test_accuracy += (depth_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        depth_lidar_mmwave_outputs = depth_lidar_mmwave_outputs.detach().cpu()
        depth_lidar_mmwave_predict_y = depth_lidar_mmwave_predict_y.detach().cpu()

        'Four modality'
        ### VK + depth + lidar + mmwave
        VK_depth_lidar_mmwave_modality_list = [True, True, True, True]
        VK_depth_lidar_mmwave_outputs = model(VK_data, depth_data,  mmwave_data, lidar_data, VK_depth_lidar_mmwave_modality_list)
        VK_depth_lidar_mmwave_outputs = VK_depth_lidar_mmwave_outputs.type(torch.FloatTensor)
        VK_depth_lidar_mmwave_outputs.to(device)
        VK_depth_lidar_mmwave_test_loss += criterion(VK_depth_lidar_mmwave_outputs,labels).item() * VK_data.size(0)
        VK_depth_lidar_mmwave_predict_y = torch.argmax(VK_depth_lidar_mmwave_outputs,dim=1).to(device)
        VK_depth_lidar_mmwave_test_accuracy += (VK_depth_lidar_mmwave_predict_y == labels.to(device)).sum().item() / labels.size(0)
        VK_depth_lidar_mmwave_outputs = VK_depth_lidar_mmwave_outputs.detach().cpu()
        VK_depth_lidar_mmwave_predict_y = VK_depth_lidar_mmwave_predict_y.detach().cpu()


    'single modality'
    ### VK
    VK_test_loss = VK_test_loss/len(tensor_loader.dataset)
    VK_test_accuracy = VK_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK',float(VK_test_loss), float(VK_test_accuracy)))
    ### depth
    depth_test_loss = depth_test_loss/len(tensor_loader.dataset)
    depth_test_accuracy = depth_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth',float(depth_test_loss), float(depth_test_accuracy)))
    ### lidar
    lidar_test_loss = lidar_test_loss/len(tensor_loader.dataset)
    lidar_test_accuracy = lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Lidar',float(lidar_test_loss), float(lidar_test_accuracy)))
    ### mmwave
    mmwave_test_loss = mmwave_test_loss/len(tensor_loader.dataset)
    mmwave_test_accuracy = mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('mmWave',float(mmwave_test_loss), float(mmwave_test_accuracy)))
    
    'dual modality'
    ### VK + depth
    VK_depth_test_loss = VK_depth_test_loss/len(tensor_loader.dataset)
    VK_depth_test_accuracy = VK_depth_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Depth',float(VK_depth_test_loss), float(VK_depth_test_accuracy)))
    ### VK + lidar
    VK_lidar_test_loss = VK_lidar_test_loss/len(tensor_loader.dataset)
    VK_lidar_test_accuracy = VK_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Lidar',float(VK_lidar_test_loss), float(VK_lidar_test_accuracy)))
    ### VK + mmwave
    VK_mmwave_test_loss = VK_mmwave_test_loss/len(tensor_loader.dataset)
    VK_mmwave_test_accuracy = VK_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+mmWave',float(VK_mmwave_test_loss), float(VK_mmwave_test_accuracy)))
    ### depth + lidar
    depth_lidar_test_loss = depth_lidar_test_loss/len(tensor_loader.dataset)
    depth_lidar_test_accuracy = depth_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+Lidar',float(depth_lidar_test_loss), float(depth_lidar_test_accuracy)))
    ### depth + mmwave
    depth_mmwave_test_loss = depth_mmwave_test_loss/len(tensor_loader.dataset)
    depth_mmwave_test_accuracy = depth_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+mmWave',float(depth_mmwave_test_loss), float(depth_mmwave_test_accuracy)))
    ### lidar + mmwave
    lidar_mmwave_test_loss = lidar_mmwave_test_loss/len(tensor_loader.dataset)
    lidar_mmwave_test_accuracy = lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Lidar+mmWave',float(lidar_mmwave_test_loss), float(lidar_mmwave_test_accuracy)))
    
    'three modality'
    ### VK + depth + lidar
    VK_depth_lidar_test_loss = VK_depth_lidar_test_loss/len(tensor_loader.dataset)
    VK_depth_lidar_test_accuracy = VK_depth_lidar_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Depth+Lidar',float(VK_depth_lidar_test_loss), float(VK_depth_lidar_test_accuracy)))
    ### VK + depth + mmwave
    VK_depth_mmwave_test_loss = VK_depth_mmwave_test_loss/len(tensor_loader.dataset)
    VK_depth_mmwave_test_accuracy = VK_depth_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Depth+mmWave',float(VK_depth_mmwave_test_loss), float(VK_depth_mmwave_test_accuracy)))
    ### VK + lidar + mmwave
    VK_lidar_mmwave_test_loss = VK_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    VK_lidar_mmwave_test_accuracy = VK_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Lidar+mmWave',float(VK_lidar_mmwave_test_loss), float(VK_lidar_mmwave_test_accuracy)))
    ### depth + lidar + mmwave
    depth_lidar_mmwave_test_loss = depth_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    depth_lidar_mmwave_test_accuracy = depth_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('Depth+Lidar+mmWave',float(depth_lidar_mmwave_test_loss), float(depth_lidar_mmwave_test_accuracy)))
    
    'four modality'
    ### VK + depth + lidar + mmwave
    VK_depth_lidar_mmwave_test_loss = VK_depth_lidar_mmwave_test_loss/len(tensor_loader.dataset)
    VK_depth_lidar_mmwave_test_accuracy = VK_depth_lidar_mmwave_test_accuracy/len(tensor_loader)
    print("modality: {}, Cross Entropy Loss: {:.8f}, Accuracy: {:.8f}".format('VK+Depth+Lidar+mmWave',float(VK_depth_lidar_mmwave_test_loss), float(VK_depth_lidar_mmwave_test_accuracy)))
    
    return

