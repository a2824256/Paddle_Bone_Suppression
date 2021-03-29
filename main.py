import paddle
import PIL.Image as image
import numpy as np
import cv2
import sp2
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

# class MSSSIMLoss(paddle.nn.layer):
#     def __init__(self):
#         super(MSSSIMLoss, self).__init__()
#
#     def forward(self, input, label):
#         msssim = _msssim(input, label)
#         loss = 1 - msssim
#         return paddle.to_tensor(loss)

# class BoneSuppressionNetwork(paddle.nn.Layer):
#     def __init__(self):
#         super(BoneSuppressionNetwork, self).__init__()
#         self.conv2d_1 = paddle.nn.Conv2D(in_channels=3, out_channels=16, kernel_size=5, padding='SAME')
#         self.relu_1 = paddle.nn.ReLU()
#         self.maxpool_1 = paddle.nn.MaxPool2D()
#         self.conv2d_2 = paddle.nn.Conv2D(in_channels=16, out_channels=32, kernel_size=5, padding='SAME')
#         self.relu_2 = paddle.nn.ReLU()
#         self.maxpool_2 = paddle.nn.MaxPool2D()
#         self.conv2d_3 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, padding='SAME')
#         self.relu_3 = paddle.nn.ReLU()
#         self.maxpool_3 = paddle.nn.MaxPool2D()
#         self.conv2d_transpose_1 = paddle.nn.Conv2DTranspose()

def _ssim(img1, img2, L = 255):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    cs_map = np.divide((2*sigma12+C2), (sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean(), cs_map.mean()

def mul_ssim(img1, img2):
    if len(img1.shape) > 2:
        ssim_arr = []
        cs_arr = []
        for i in range(0, img1.shape[2]):
            ssim_val, cs = _ssim(img1[:, :, i], img2[:, :, i])
            ssim_arr.append(ssim_val)
            cs_arr.append(cs)
        return np.array(ssim_arr).mean(), np.array(cs_arr).mean()
    else:
        return _ssim(img1, img2)

def _msssim(img1, img2):
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    level = len(weight)
    mcs_array = []
    ssim_array = []
    img1 = np.array([[img1]])
    img2 = np.array([[img2]])
    for i in range(level):
        ssim, cs = mul_ssim(img1[0][0], img2[0][0])
        mcs_array.append(cs.mean())
        ssim_array.append(ssim)
        with fluid.dygraph.guard():
            padding = (img1.shape[2]%2, img2.shape[3]%2)
            # print('padding:', padding)
            pool2d_1 = fluid.dygraph.Pool2D(pool_type='avg', pool_size=2, use_cudnn=False, pool_padding=padding)
            pool2d_2 = fluid.dygraph.Pool2D(pool_type='avg', pool_size=2, use_cudnn=False, pool_padding=padding)
            filtered_im1 = pool2d_1(to_variable(img1))
            filtered_im2 = pool2d_2(to_variable(img2))
            # filtered_im1 = ndimage.filters.convolve(img1, downsample_filter,
            #                                         mode='reflect')
            # filtered_im2 = ndimage.filters.convolve(img2, downsample_filter,
            #                                         mode='reflect')
            # filtered_im1 = cv2.filter2D(img1, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
            # filtered_im2 = cv2.filter2D(img2, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
            # img1 = filtered_im1[::2, ::2]
            # img2 = filtered_im2[::2, ::2]
            img1 = filtered_im1.numpy()
            img2 = filtered_im2.numpy()

    overall_mssim = np.prod(np.power(mcs_array[:level - 1], weight[:level - 1])) * (
                ssim_array[level - 1] ** weight[level - 1])
    return overall_mssim


def transfer_16bit_to_8bit(img_matrix):
    min_16bit = np.min(img_matrix)
    max_16bit = np.max(img_matrix)
    return np.array(np.rint(255 * ((img_matrix - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)


def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]


def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n-1)


if __name__=="__main__":
    origin_path = './origin.png'
    # origin_path = './3.jpg'
    pred_path = './pred.png'
    # pred_path = './3.jpg'
    img_ori = image.open(origin_path)
    img_pred = image.open(pred_path)
    X = np.array(img_pred)
    Y = np.array(img_ori).astype('float32')
    X = transfer_16bit_to_8bit(X).astype('float32')
    print(_msssim(X, Y))