import torch

class SLDA(torch.nn.Module):
    # Achieves the same accuracy as the online variant and but is wayy faster. Tested against this code for the online variant: https://github.com/tyler-hayes/Deep_SLDA/blob/master/SLDA_Model.py
    def __init__(self, input_shape, num_classes, shrinkage_param=1e-4):
        super(SLDA, self).__init__()
        self.input_shape = input_shape
        self.shrinkage_param = shrinkage_param
        self.muK = torch.zeros((num_classes, input_shape)).float()
        self.Sigma = torch.ones((input_shape, input_shape)).float()
            
    def predict(self, X, return_probas=False):
        X = X.float()

        with torch.no_grad():
            M = self.muK.transpose(1, 0).float()
            W = torch.linalg.lstsq((1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).float(), M).solution
            c = 0.5 * torch.sum(M * W, dim=0)
            scores = torch.matmul(X, W) - c

        # return predictions or probabilities
        if not return_probas:
            return scores.argmax(dim=1)
        else:
            return torch.softmax(scores, dim=1)

    def fit(self, X, y):
        X = X.float()
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)

        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float()
