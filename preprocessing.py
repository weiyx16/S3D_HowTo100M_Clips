import torch

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor

class Preprocessing(object):

    def __init__(self, framenum=16):
        self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])
        self.framenum = framenum

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = torch.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return torch.cat((tensor, z), 0)

    def __call__(self, tensor, ):
        tensor = self._zero_pad(tensor, self.framenum)
        tensor = self.norm(tensor)
        tensor = tensor.view(-1, self.framenum, 3, tensor.size(-2), tensor.size(-1))
        tensor = tensor.transpose(1, 2)
        return tensor