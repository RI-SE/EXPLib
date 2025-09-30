########## Forked from https://github.com/maxidl/MMD-critic ###

from pathlib import Path
import torch


def default_gamma(X:torch.Tensor):
    gamma = 1.0 / X.shape[1]
    print(f'Setting default gamma={gamma}')
    return gamma


def rbf_kernel(X:torch.Tensor, gamma:float=None):
    assert len(X.shape) == 2

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.cdist(X, X)
    K.fill_diagonal_(0) # avoid floating point error
    K.pow_(2)
    K.mul_(-gamma)
    K.exp_()
    return K


def local_rbf_kernel(X:torch.Tensor, y:torch.Tensor, gamma:float=None):
    # todo make final representation sparse (optional)
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert torch.all(y == y.sort()[0]), 'This function assumes the dataset is sorted by y'

    if gamma is None:
        gamma = default_gamma(X)
    K = torch.zeros((X.shape[0], X.shape[0]))
    y_unique = y.unique()
    for i in range(y_unique[-1] + 1): # compute kernel blockwise for each class
        ind = torch.where(y == y_unique[i])[0]
        start = ind.min()
        end = ind.max() + 1
        K[start:end, start:end] = rbf_kernel(X[start:end, :], gamma=gamma)
    return K


def change_gamma(K:torch.Tensor, old_gamma:float, new_gamma:float):
    assert K.shape[0] == K.shape[1]
    K.log_()
    K.div_(-old_gamma)
    K.mul_(-new_gamma)
    K.exp_()
    return K

class Dataset:
    def __init__(self, X: torch.Tensor, y:torch.Tensor) -> None:
        assert X.dtype == torch.float
        assert y.dtype == torch.long
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]

        self.sort_indices = y.argsort()
        self.X = X[self.sort_indices, :]
        self.y = y[self.sort_indices]

    def compute_rbf_kernel(self, gamma:float=None):
        self.K = rbf_kernel(self.X, gamma)
        self.gamma = gamma
        self.kernel_type = 'global'

    def compute_local_rbf_kernel(self, gamma:float=None):
        self.K = local_rbf_kernel(self.X, self.y, gamma)
        self.gamma = gamma
        self.kernel_type = 'local'

    def set_gamma(self, gamma:float):
        if self.K is None:
            raise AttributeError('Kernel K has not been computed yet.')
        change_gamma(self.K, self.gamma, gamma)
        self.gamma = gamma
    
    def dump_kernel(self, dest:Path):
        torch.save(self.K, dest)

    def load_kernel(self, src:Path):
        K = torch.load(src)
        assert self.K.shape[0] == self.X.shape[0] and self.K.shape[0] == self.K.shape[1]
        self.K = K


def select_prototypes(K:torch.Tensor, num_prototypes:int):
    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    colsum = 2 * K.sum(0) / num_samples
    is_selected = torch.zeros_like(sample_indices)
    selected = sample_indices[is_selected > 0]

    for i in range(num_prototypes):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]

        if selected.shape[0] == 0:
            s1 -= K.diagonal()[candidate_indices].abs()
        else:
            temp = K[selected, :][:, candidate_indices]
            s2 = temp.sum(0) * 2 + K.diagonal()[candidate_indices]
            s2 /= (selected.shape[0] + 1)
            s1 -= s2

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1
        selected = sample_indices[is_selected > 0]

    selected_in_order = selected[is_selected[is_selected > 0].argsort()]
    return selected_in_order


def select_criticisms(K:torch.Tensor, prototype_indices:torch.Tensor, num_criticisms:int, regularizer=None):
    prototype_indices = prototype_indices.clone()
    available_regularizers = {None, 'logdet', 'iterative'}
    assert regularizer in available_regularizers, f'Unknown regularizer: "{regularizer}". Available regularizers: {available_regularizers}'

    sample_indices = torch.arange(0, K.shape[0])
    num_samples = sample_indices.shape[0]

    is_selected = torch.zeros_like(sample_indices)
    is_selected[prototype_indices] = num_criticisms + 1 # is_selected > 0 indicates either selected (1 to num_criticisms) or prototype (if num_criticisms +1)
    selected = sample_indices[is_selected > 0]

    colsum = K.sum(0) / num_samples
    inverse_of_prev_selected = None
    for i in range(num_criticisms):
        candidate_indices = sample_indices[is_selected == 0]
        s1 = colsum[candidate_indices]

        temp = K[prototype_indices, :][:, candidate_indices]
        s2 = temp.sum(0)
        s2 /= prototype_indices.shape[0]
        s1 -= s2
        s1.abs_()

        if regularizer == 'logdet':
            if inverse_of_prev_selected is not None: # first call has been made already
                temp = K[selected, :][:, candidate_indices]
                temp2 = inverse_of_prev_selected.mm(temp) # torch.mm replaces np.dot
                reg = temp2 * temp
                regcolsum = reg.sum(0)
                reg = (K.diagonal()[candidate_indices] - regcolsum).abs().log()
                s1 += reg
            else:
                s1 -= K.diagonal()[candidate_indices].abs().log()

        best_sample_index = candidate_indices[s1.argmax()]
        is_selected[best_sample_index] = i + 1

        selected = sample_indices[(is_selected > 0) & (is_selected != (num_criticisms + 1))]

        if regularizer == 'iterative':
            prototype_indices = torch.cat([prototype_indices, best_sample_index.unsqueeze(0)])

        if regularizer == 'logdet':
            KK = K[selected,:][:,selected]
            inverse_of_prev_selected = torch.inverse(KK) # shortcut

    selected_in_order = selected[is_selected[(is_selected > 0) & (is_selected != (num_criticisms + 1))].argsort()]
    return selected_in_order
