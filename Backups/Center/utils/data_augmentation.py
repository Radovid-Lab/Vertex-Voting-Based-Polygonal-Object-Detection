import random
import torch as th

class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, x, y=None):
        x = x.numpy()
        if y is not None:
            for i in y:
                y[i]=y[i].numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    for i in y:
                        y[i] = y[i].swapaxes(2, 0)
                        if i == 'yvector':
                            y[i] = -y[i][::-1, ...]
                        else:
                            y[i] = y[i][::-1, ...]
                        y[i] = y[i].swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    for i in y:
                        y[i] = y[i].swapaxes(1, 0)
                        if i == 'xvector':
                            y[i] = -y[i][::-1, ...]
                        else:
                            y[i] = y[i][::-1, ...]
                        y[i] = y[i].swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return th.from_numpy(x.copy())
        else:
            for i in y:
                y[i]=th.from_numpy(y[i].copy())
            return th.from_numpy(x.copy()),y