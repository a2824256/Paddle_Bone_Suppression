import paddle
import PIL.Image as image
import numpy as np
import ssim
from matplotlib import pyplot as plt
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
from skimage.metrics import structural_similarity

# class MSSSIMLoss(paddle.nn.layer):
#     def __init__(self):
#         super(MSSSIMLoss, self).__init__()
#
#     def forward(self, input, label):
#         return 0



origin_path = './origin.png'
pred_path = './pred.png'

def transfer_16bit_to_8bit(img_matrix):
    min_16bit = np.min(img_matrix)
    max_16bit = np.max(img_matrix)
    return np.array(np.rint(255 * ((img_matrix - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)

# class AELikeModel(paddle.nn.layer):
#     def __init__(self):
#         super(AELikeModel, self).__init__()


if __name__=="__main__":

    img_ori = image.open(origin_path)
    img_pred = image.open(pred_path)
    X = np.array(img_ori)
    Y = X
    # Y = transfer_16bit_to_8bit(np.array(img_pred))
    ssim_obj = ssim.SSIM(crop_border=0)
    ssim_obj.update(X, Y)
    print(ssim_obj.accumulate())