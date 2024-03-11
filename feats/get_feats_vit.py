import torch, timm
import torchvision
import numpy as np
from torchvision import transforms, datasets

def get_dataset(dataset, batchsize, transforms_list=None):
    assert(dataset in ['CIFAR100', 'imagenet-r','imagenet-a','cub','omnibenchmark','vtab','cars'])
    print('==> Loading datasets..')
    if dataset == 'CIFAR100':
        transforms_list = transforms.Compose([transforms.Resize(224, interpolation=3, antialias=True), transforms.ToTensor()])
    else:
        transforms_list = transforms.Compose([transforms.Resize(256, interpolation=3, antialias=True), transforms.CenterCrop(224), transforms.ToTensor()])
    
   if dataset in ['CIFAR10', 'CIFAR100']:
        dset = getattr(torchvision.datasets, dataset)
        kwargs_train = {'train': True, 'download': True}
        kwargs_test = {'train': False, 'download': True}
        train_data = dset('../data/', transform=transforms_list, **kwargs_train)
        test_data = dset('../data/', transform=transforms_list, **kwargs_test)
    elif dataset in ['imagenet-r', 'imagenet-a', 'cub', 'omnibenchmark', 'vtab', 'cars']:
        train_data = datasets.ImageFolder('../data/'+dataset+'/train/', transform=transforms_list)
        test_data = datasets.ImageFolder('../data/'+dataset+'/test/', transform=transforms_list)

    trainlen, testlen = len(train_data), len(test_data)
    print(trainlen, testlen)

    trainloader =  torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    testloader =  torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return trainloader, testloader, trainlen, testlen


def extract_feats(model, loader, batchsize, featsize, num_samples, expname):
    print('==> Extracting features..')
    model.cuda()
    model.eval()

    labelarr, featarr, flipped_featarr = np.zeros(num_samples, dtype='u2'), np.zeros((num_samples, featsize),dtype=np.float32), np.zeros((num_samples, featsize),dtype=np.float32)

    with torch.inference_mode():
        for count, (image, label) in enumerate(loader):
            idx = (np.ones(batchsize)*count*batchsize+np.arange(batchsize)).astype(int)
            idx = idx[:label.shape[0]]
            image = image.cuda(non_blocking=True)
            # Flip image tensor horizontally

            feat = model(image)
            labelarr[idx] = label.numpy()
            featarr[idx] = feat.cpu().numpy()

            image_flipped = image.flip(dims=[3])
            feat_flipped = model(image_flipped)
            flipped_featarr[idx] = feat_flipped.cpu().numpy()
        np.save('./'+expname+'_features_flipped.npy', flipped_featarr)
        np.save('./'+expname+'_features.npy', featarr)
        np.save('./'+expname+'_labels.npy', labelarr)
    return

if __name__ == '__main__':
    model.cuda()
    model.eval()
    featsize = 768
    batchsize = 512

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    
    trainloader, testloader, trainlen, testlen = get_dataset('CIFAR100', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_cifar100_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_cifar100_test')

    trainloader, testloader, trainlen, testlen = get_dataset('imagenet-r', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_imagenet-r_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_imagenet-r_test')

    trainloader, testloader, trainlen, testlen = get_dataset('imagenet-a', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_imagenet-a_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_imagenet-a_test')

    trainloader, testloader, trainlen, testlen = get_dataset('cub', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_cub_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_cub_test')

    trainloader, testloader, trainlen, testlen = get_dataset('omnibenchmark', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_omnibenchmark_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_omnibenchmark_test')

    trainloader, testloader, trainlen, testlen = get_dataset('vtab', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_vtab_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_vtab_test')

    trainloader, testloader, trainlen, testlen = get_dataset('cars', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi1k_cars_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi1k_cars_test')
    

    # ADD YOUR OWN DATASETS HERE

    model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    
    trainloader, testloader, trainlen, testlen = get_dataset('CIFAR100', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_cifar100_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_cifar100_test')

    trainloader, testloader, trainlen, testlen = get_dataset('imagenet-r', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_imagenet-r_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_imagenet-r_test')

    trainloader, testloader, trainlen, testlen = get_dataset('imagenet-a', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_imagenet-a_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_imagenet-a_test')

    trainloader, testloader, trainlen, testlen = get_dataset('cub', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_cub_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_cub_test')

    trainloader, testloader, trainlen, testlen = get_dataset('omnibenchmark', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_omnibenchmark_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_omnibenchmark_test')

    trainloader, testloader, trainlen, testlen = get_dataset('vtab', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_vtab_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_vtab_test')

    trainloader, testloader, trainlen, testlen = get_dataset('cars', batchsize)
    extract_feats(model, trainloader, batchsize, featsize, trainlen, 'vitbi21k_cars_train')
    extract_feats(model, testloader, batchsize, featsize, testlen, 'vitbi21k_cars_test')

    # ADD YOUR OWN DATASETS HERE
