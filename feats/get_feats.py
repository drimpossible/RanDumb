import numpy as np
import torchvision, glob
from PIL import Image
from datasets import load_dataset

def get_combined_X(a, b):
    c = np.empty((a.shape[0] + b.shape[0], a.shape[1], a.shape[2], a.shape[3]), dtype=a.dtype)
    c[0::2,:,:,:] = a
    c[1::2,:,:,:] = b
    return c

def get_combined_y(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def process_data(X, y, mean, var, color=True, flip=True):
    if color:
        X = np.transpose(X, (0, 3, 1, 2))

    idx = []
    for i in range(len(np.unique(y))):
        idx.append(np.where(y == i)[0])
    idx = np.concatenate(idx, axis=0)

    X = X[idx]
    y = y[idx]   

    X = X.astype(np.float32)/255
    X = (X - mean)/var
    
    if flip:
        X_flip = X[:, :, :, ::-1]
        X_full = get_combined_X(X, X_flip)
        y_full = get_combined_y(y, y)
        X = X.reshape(X.shape[0], -1)
        X_full = X_full.reshape(X_full.shape[0], -1)
        return X, y, X_full, y_full
    
    X = X.reshape(X.shape[0], -1)
    return X, y

mean_list = {
        'MNIST':(0.1307,),
        'CIFAR10':(0.4914, 0.4822, 0.4465),
        'CIFAR100':(0.5071, 0.4867, 0.4408),
        'TinyImagenet':(0.4802, 0.4481, 0.3975),
        'miniImagenet':(0.485, 0.456, 0.406),
}

var_list = {
        'MNIST':(0.3081,),
        'CIFAR10':(0.2470, 0.2435, 0.2616),
        'CIFAR100':(0.2675, 0.2565, 0.2761),
        'TinyImagenet':(0.2302, 0.2265, 0.2262),
        'miniImagenet': (0.229, 0.224, 0.225),
}


if __name__ == '__main__':
    d = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    train_X = d.data
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['CIFAR100'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(var_list['CIFAR100'])[np.newaxis, :, np.newaxis, np.newaxis]
    train_X, train_y, train_X_combined, train_y_combined = process_data(train_X, train_y, mean, var, color=True, flip=True)
    np.save(f"cifar100_train_features.npy", train_X)
    np.save(f"cifar100_train_labels.npy", train_y)
    np.save(f"cifar100_train_features_combined.npy", train_X_combined)
    np.save(f"cifar100_train_labels_combined.npy", train_y_combined)

    d = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    test_X = d.data
    test_y = np.array(d.targets)
    test_X, test_y = process_data(test_X, test_y, mean, var, color=True, flip=False)
    np.save(f"cifar100_test_features.npy", test_X)
    np.save(f"cifar100_test_labels.npy", test_y)

    d = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    train_X = d.data
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['CIFAR10'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(var_list['CIFAR10'])[np.newaxis, :, np.newaxis, np.newaxis]
    train_X, train_y, train_X_combined, train_y_combined = process_data(train_X, train_y, mean, var, color=True, flip=True)
    np.save(f"cifar10_train_features.npy", train_X)
    np.save(f"cifar10_train_labels.npy", train_y)
    np.save(f"cifar10_train_features_combined.npy", train_X_combined)
    np.save(f"cifar10_train_labels_combined.npy", train_y_combined)

    d = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    test_X = d.data
    test_y = np.array(d.targets)
    test_X, test_y = process_data(test_X, test_y, mean, var, color=True, flip=False)
    np.save(f"cifar10_test_features.npy", test_X)
    np.save(f"cifar10_test_labels.npy", test_y)

    d = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    train_X = d.data.numpy()
    train_y = np.array(d.targets)
    mean, var = np.array(mean_list['MNIST'])[np.newaxis, :, np.newaxis], np.array(var_list['MNIST'])[np.newaxis, :, np.newaxis]
    train_X, train_y = process_data(train_X, train_y, mean, var, color=False, flip=False)
    np.save(f"mnist_train_features.npy", train_X)
    np.save(f"mnist_train_labels.npy", train_y)

    d = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    test_X = d.data.numpy()
    test_y = np.array(d.targets)
    test_X, test_y = process_data(test_X, test_y, mean, var, color=False, flip=False)
    np.save(f"mnist_test_features.npy", test_X)
    np.save(f"mnist_test_labels.npy", test_y)

    d = load_dataset('Maysee/tiny-imagenet', split='train')
    train_X_smol, train_y_smol = [], []

    for i in range(len(d)):
        im = d[i]['image']
        im = im.convert('RGB')
        im_smol = im.resize((32, 32))
        train_X_smol.append(np.array(im_smol)[np.newaxis, ...])
        train_y_smol.append(d[i]['label'])

    train_X_smol = np.concatenate(train_X_smol, axis=0)
    train_y_smol = np.array(train_y_smol)
    mean, var = np.array(mean_list['TinyImagenet'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(var_list['TinyImagenet'])[np.newaxis, :, np.newaxis, np.newaxis]

    train_X_smol, train_y_smol, train_X_smol_combined, train_y_smol_combined = process_data(train_X_smol, train_y_smol, mean, var, color=True, flip=True)
    np.save(f"tinyimagenet_smol_train_features.npy", train_X_smol)
    np.save(f"tinyimagenet_smol_train_labels.npy", train_y_smol)
    np.save(f"tinyimagenet_smol_train_features_combined.npy", train_X_smol_combined)
    np.save(f"tinyimagenet_smol_train_labels_combined.npy", train_y_smol_combined)

    d = load_dataset('Maysee/tiny-imagenet', split='valid')
    test_X_smol, test_y_smol = [], []

    for i in range(len(d)):
        im = d[i]['image']
        im = im.convert('RGB')
        im_smol = im.resize((32, 32))
        test_X_smol.append(np.array(im_smol)[np.newaxis, ...])
        test_y_smol.append(d[i]['label'])

    test_X_smol = np.concatenate(test_X_smol, axis=0)
    test_y_smol = np.array(test_y_smol)

    test_X_smol, test_y_smol = process_data(test_X_smol, test_y_smol, mean, var, color=True, flip=False)
    np.save(f"tinyimagenet_smol_test_features.npy", test_X_smol)
    np.save(f"tinyimagenet_smol_test_labels.npy", test_y_smol)

    miniimagenet_classes = ["n01532829", "n01558993", "n01704323", "n01749939", "n01770081", "n01843383", "n01855672", "n01910747", "n01930112", "n01981276", "n02074367", "n02089867", "n02091244", "n02091831", "n02099601", "n02101006", "n02105505", "n02108089", "n02108551", "n02108915", "n02110063", "n02110341", "n02111277", "n02113712", "n02114548", "n02116738", "n02120079", "n02129165", "n02138441", "n02165456", "n02174001", "n02219486", "n02443484", "n02457408", "n02606052", "n02687172", "n02747177", "n02795169", "n02823428", "n02871525", "n02950826", "n02966193", "n02971356", "n02981792", "n03017168", "n03047690", "n03062245", "n03075370", "n03127925", "n03146219", "n03207743", "n03220513", "n03272010", "n03337140", "n03347037", "n03400231", "n03417042", "n03476684", "n03527444", "n03535780", "n03544143", "n03584254", "n03676483", "n03770439", "n03773504", "n03775546", "n03838899", "n03854065", "n03888605", "n03908618", "n03924679", "n03980874", "n03998194", "n04067472", "n04146614", "n04149813", "n04243546", "n04251144", "n04258138", "n04275548", "n04296562", "n04389033", "n04418357", "n04435653", "n04443257", "n04509417", "n04515003", "n04522168", "n04596742", "n04604644", "n04612504", "n06794110", "n07584110", "n07613480", "n07697537", "n07747607", "n09246464", "n09256479", "n13054560", "n13133613"]
    train_X_smol, train_y_smol = [], []
    test_X_smol, test_y_smol = [], []

    for i in range(len(miniimagenet_classes)):
        path = '<path_to_imagenet1k>/Imagenet/train/' + miniimagenet_classes[i] + '/'
        # Load all images in this folder sequentially using glob
        for filename in glob.glob(path + '*.JPEG'):
            im = Image.open(filename)
            im = im.convert('RGB')
            im_smol = im.resize((32, 32))
            train_X_smol.append(np.array(im_smol)[np.newaxis, ...])
            train_y_smol.append(i)

        path = '<path_to_imagenet1k>/Imagenet/val/' + miniimagenet_classes[i] + '/'
        # Load all images in this folder sequentially using glob
        for filename in glob.glob(path + '*.JPEG'):
            im = Image.open(filename)
            im = im.convert('RGB')
            im_smol = im.resize((32, 32))
            test_X_smol.append(np.array(im_smol)[np.newaxis, ...])
            test_y_smol.append(i)

    train_X_smol = np.concatenate(train_X_smol, axis=0)
    train_y_smol = np.array(train_y_smol)
    test_X_smol = np.concatenate(test_X_smol, axis=0)
    test_y_smol = np.array(test_y_smol)
    mean, var = np.array(mean_list['miniImagenet'])[np.newaxis, :, np.newaxis, np.newaxis], np.array(var_list['miniImagenet'])[np.newaxis, :, np.newaxis, np.newaxis]

    train_X_smol, train_y_smol, train_X_smol_combined, train_y_smol_combined = process_data(train_X_smol, train_y_smol, mean, var, color=True, flip=True)
    np.save(f"miniimagenet_smol_train_features.npy", train_X_smol)
    np.save(f"miniimagenet_smol_train_labels.npy", train_y_smol)
    np.save(f"miniimagenet_smol_train_features_combined.npy", train_X_smol_combined)
    np.save(f"miniimagenet_smol_train_labels_combined.npy", train_y_smol_combined)

    test_X_smol, test_y_smol = process_data(test_X_smol, test_y_smol, mean, var, color=True, flip=False)
    np.save(f"miniimagenet_smol_test_features.npy", test_X_smol)
    np.save(f"miniimagenet_smol_test_labels.npy", test_y_smol)
