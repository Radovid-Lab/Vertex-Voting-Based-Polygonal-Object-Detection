import numpy as np
import torch
# functions for gaussian augmentation

def gaussian2D(shape, sigma=1):
    '''
    Generate 2D gaussian
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmaps, radius, k=1, decreasing=True):
    '''
    Apply 2D gaussian
    :param heatmap: feature map on which gaussian will be applied [num_classes,image_size[0],image_size[1]]
    :param center: center of gaussian
    :param radius: radius of gaussian
    :param k: value
    '''
    diameter = 2 * radius + 1
    gaussian = torch.from_numpy(gaussian2D((diameter, diameter), sigma=diameter / 6)).float()

    height, width = heatmaps.shape[1:]

    for channel in range(heatmaps.shape[0]):
        heatmap=heatmaps[channel]
        centers=torch.nonzero(heatmap,as_tuple =False)
        for center in centers:
            x, y = center
            top, bottom = min(x, radius), min(height - x, radius + 1)
            left, right = min(y, radius), min(width - y, radius + 1)
            masked_heatmap = heatmap[x - top:x + bottom, y - left:y + right]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            assert masked_heatmap.shape == masked_gaussian.shape, 'mask shape in draw_gaussian are not equal'
            if decreasing:
                if k > 0:
                    torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
                else:
                    torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            else:
                if k > 0:
                    torch.max(masked_heatmap, np.ones_like(masked_heatmap) * k, out=masked_heatmap)
                else:
                    torch.max(masked_heatmap, np.ones_like(masked_heatmap) * k, out=masked_heatmap)
    return heatmaps