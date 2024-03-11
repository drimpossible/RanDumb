import os, argparse, logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.kernel_approximation import RBFSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'tinyimagenet_smol', 'miniimagenet_smol', 'vitbi1k_cars',  'vitbi21k_cars',  'vitbi1k_cifar100', 'vitbi1k_imagenet-r', 'vitbi1k_imagenet-a', 'vitbi1k_cub', 'vitbi1k_omnibenchmark', 'vitbi1k_vtab', 'vitbi1k_cars', 'vitbi21k_cifar100', 'vitbi21k_imagenet-r', 'vitbi21k_imagenet-a', 'vitbi21k_cub', 'vitbi21k_omnibenchmark', 'vitbi21k_vtab'], help='Dataset')
    parser.add_argument('--model', type=str, default='SLDA', choices=['SLDA', 'NCM'], help='Model')
    parser.add_argument('--augment', action='store_true', help='Use RandomFlip Augmentation')
    parser.add_argument('--embed', action='store_true', help='Use embedding projection')
    parser.add_argument('--embed_mode', type=str, default='RanDumb', choices=['RanDumb','RanPAC'], help='Choice of embedding')
    parser.add_argument('--embed_dim', type=int, default=10000, help='Embedding dimension')
    # Default args
    parser.add_argument('--feature_path', type=str, default='../feats/', help='Path to features')
    parser.add_argument('--log_dir', type=str, default='../logs/', help='Path to logs')
    args = parser.parse_args()
    return args


def get_logger(folder, name):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    
    # file logger
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fh = logging.FileHandler(os.path.join(folder, name+'_checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
    

if __name__ == '__main__':
    args = parse_args()
    exp_name= f"{args.dataset}_{args.model}_{args.augment}_{args.embed}"
    if args.embed:
        exp_name += f"_{args.embed_mode}_{args.embed_dim}"

    console_logger = get_logger(folder=args.log_dir, name=exp_name)
    console_logger.debug(args)

    if args.augment:
        train_X = np.load(os.path.join(args.feature_path, f"{args.dataset}_train_features_combined.npy"))
        train_y = np.load(os.path.join(args.feature_path, f"{args.dataset}_train_labels_combined.npy"))
    else:
        train_X = np.load(os.path.join(args.feature_path, f"{args.dataset}_train_features.npy"))
        train_y = np.load(os.path.join(args.feature_path, f"{args.dataset}_train_labels.npy"))
    
    args.num_classes = len(np.unique(train_y))
    test_X = np.load(os.path.join(args.feature_path, f"{args.dataset}_test_features.npy"))
    test_y = np.load(os.path.join(args.feature_path, f"{args.dataset}_test_labels.npy"))

    train_y = train_y.astype(np.int32)
    test_y = test_y.astype(np.int32)
    class_ordering = np.arange(args.num_classes) # Insert your fancy class ordering here
    idx = []
    for i in range(args.num_classes):
        idx.append(np.where(train_y == class_ordering[i])[0])
    idx = np.concatenate(idx, axis=0)
    train_X = train_X[idx]
    train_y = train_y[idx]
    
    if args.embed_mode == 'RanPAC':
        W = np.random.randn(train_X.shape[1],args.embed_dim)
        train_X = np.maximum(0, np.matmul(train_X, W))
        test_X = np.maximum(0, np.matmul(test_X, W))
    elif args.embed_mode == 'RanDumb':
        embedder = RBFSampler(gamma='scale', n_components=args.embed_dim)
        embedder.fit(train_X) # The scikit function ignores data passed to it, using on the input dimensions. We are not fitting anything here with data.
        train_X = embedder.transform(train_X)
        test_X = embedder.transform(test_X)
                            
    if args.model == 'SLDA':
        oa = OAS(assume_centered=False) # Very sample-efficient shrinkage estimator 
        model = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=oa) # Main difference between original paper code and here. Faster, easier to play but roughly equivalent to the online version: https://github.com/tyler-hayes/Deep_SLDA/blob/master/SLDA_Model.py with better-set shrinkage. Tested against original online code with hparam search for shrinkage, returns similar results (\pm 0.8)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        matrix = confusion_matrix(test_y, preds) # Some datasets are imbalanced, so we calculate per class accuracy to get the average incremental accuracy
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)
        
    elif args.model == 'NCM':
        model = NearestCentroid(metric='cosine')
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        matrix = confusion_matrix(test_y, preds)
        acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
        acc = np.mean(acc_per_class)

    logger_out = f"Test accuracy\t{acc}\tDataset\t{args.dataset}\tModel\t{args.model}\tAugment\t{args.augment}\tEmbed\t{args.embed}"
    if args.embed:
        logger_out += f"\tEmbed_mode\t{args.embed_mode}\tEmbed_dim\t{args.embed_dim}"
    console_logger.info(logger_out)
