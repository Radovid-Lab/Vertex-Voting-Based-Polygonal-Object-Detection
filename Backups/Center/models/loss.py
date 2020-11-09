import torch
from conf.config import Config

class HeatmapLossSL1(torch.nn.Module):
    '''
    ratio is to control the contribution of positive and negative loss.
    eg. ratio=0.1 means the contribution of negative is 0.1 times of positive
    '''
    def __init__(self,ratio=0.01):
        super(HeatmapLossSL1, self).__init__()
        assert ratio<=1,'HeatmapLossSL1 ratio should be smaller than 1'
        self.ratio=ratio
        self.SmoothL1=torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='none')

    def forward(self, pred, gt, mean=True):
        assert pred.size() == gt.size()
        maskpos=(gt!=0).float()
        maskneg=(gt==0).float()*self.ratio
        l=maskpos*self.SmoothL1(pred,gt)+maskneg*self.SmoothL1(pred,gt)
        if mean:
            return l.mean()
        else:
            return l.sum()

class HeatmapLossMSE(torch.nn.Module):
    '''
    when mean is true the loss is mse. otherwise return sum square loss
    '''
    def __init__(self,config:Config,weight=1.0):
        super(HeatmapLossMSE, self).__init__()
        self.weight=weight
        self.config=config

    def forward(self, pred, gt, mean):
        assert pred.size() == gt.size()
        l = (pred - gt)**2
        mask=gt!=0
        mask=(mask.float()*self.weight+torch.ones(*l.shape))
        l=l*mask
        if mean:
            return l.mean()
        else:
            return l.sum()

# weighted bce loss
## need rewriting
def weightedBCE(positiveWeight: int=1):
    '''
    weighted BCE loss
    :param positiveWeight: The weight put on positive pixels
    :return: weighted BCE function
    '''
    def weighted_binary_cross_entropy(output, target, posweights=positiveWeight,reduce=True,average=True):
        '''
        :param reduce: indicate whether reduce to a scalar
        :param average: indicate average or sum up the loss feature map
        '''
        if output.size()!=target.size():
            raise ValueError(f'predicted size {output.size()} is not identical with target size {target.size()}')
        loss = -posweights * target * output.log() - (1 - target) * (1 - output).log()
        if not reduce:
            return loss
        elif average:
            return loss.mean()
        else:
            return loss.sum()
    return weighted_binary_cross_entropy


# focal loss
class FocalLoss(torch.nn.Module):
    def __init__(self,gamma=2,alpha=0.75):
        '''
            γ: is a prefixed positive scalar value and
            α: is a prefixed value between 0 and 1 to balance
            the positive labeled samples and negative labeled samples
        '''
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.alpha=alpha


    def forward(self, pred, gt, mean):
        assert pred.size() == gt.size()
        tisone=-(self.alpha*(gt!=0).float())*((torch.ones_like(pred)-pred)**self.gamma)*torch.log(pred)
        tiszero=-((1-self.alpha)*(gt==0).float())*(pred**self.gamma)*torch.log(torch.ones_like(pred)-pred)
        l=tisone+tiszero
        if mean:
            return l.mean()
        else:
            return l.sum()