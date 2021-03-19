import cv2
import numpy as np
import paddle

class PSNR(paddle.metric.Metric):
    def __init__(self, crop_border, input_order='HWC', test_y_channel=False):
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.reset()

    def reset(self):
        self.results = []

    def update(self, preds, gts):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        for pred, gt in zip(preds, gts):
            value = calculate_psnr(pred, gt, self.crop_border, self.input_order,
                                   self.test_y_channel)
            self.results.append(value)

    def accumulate(self):
        if len(self.results) <= 0:
            return 0.
        return np.mean(self.results)

    def name(self):
        return 'PSNR'


class SSIM(PSNR):
    def update(self, preds, gts):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        for pred, gt in zip(preds, gts):
            value = calculate_ssim(pred, gt, self.crop_border, self.input_order,
                                   self.test_y_channel)
            self.results.append(value)

    def name(self):
        return 'SSIM'

class MSSSIM(SSIM):
    def update(self, preds, gts):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        if not isinstance(gts, (list, tuple)):
            gts = [gts]

        for pred, gt in zip(preds, gts):
            value = calculate_msssim(pred, gt, self.crop_border, self.input_order,
                                   self.test_y_channel)

            self.results.append(value)

    def name(self):
        return 'MSSSIM'

def calculate_msssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    _, l_list, c_list, s_list = calculate_ssim(img1, img2, crop_border, input_order, test_y_channel)

    return 0

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = img1.copy().astype('float32')
    img2 = img2.copy().astype('float32')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

# ssim的运算
def _ssim(img1, img2, L):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    C3 = C2/2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # ux
    ux = img1.mean()
    # uy
    uy = img2.mean()
    # ux^2
    ux_sq = ux**2
    # uy^2
    uy_sq = uy**2
    # ux*uy
    uxuy = ux * uy
    # ox、oy方差计算
    ox_sq = img1.var()
    oy_sq = img2.var()
    print("ox_sq:", ox_sq, ",oy_sq:", oy_sq)
    ox = np.sqrt(ox_sq)
    oy = np.sqrt(oy_sq)
    print("ox:", ox, ",oy:", oy)
    oxoy = ox * oy
    oxy = np.sum(np.mean((img1 - ux) * (img2 - uy)))
    print('ux:', ux, ',uy:', uy, ', ux_sq:', ux_sq, ',uy_sq:', uy_sq, ',uxuy:', uxuy, 'oxy:', oxy, 'oxoy:', oxoy)
    L = (2 * uxuy + C1) / (ux_sq + uy_sq + C1)
    C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
    S = (oxy + C3) / (oxoy + C3)
    ssim = L * C * S
    print('ssim:', ssim, ",L:", L, ",C:", C, ",S:", S)
    return ssim, L, C, S
    # return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    img1 = img1.copy().astype('float32')[..., ::-1]
    img2 = img2.copy().astype('float32')[..., ::-1]
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    l_list = []
    c_list = []
    s_list = []
    for i in range(img1.shape[2]):
        res, l, c, s = _ssim(img1[..., i], img2[..., i], 255)
        ssims.append(res)
        l_list.append(l)
        c_list.append(c)
        s_list.append(s)
    return np.array(ssims).mean(), l_list, c_list, s_list


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.
    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype

    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    return out_img


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.
    The RGB version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype

    if img_type != np.uint8:
        img *= 255.

    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) / 255. + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]

    if img_type != np.uint8:
        out_img /= 255.
    else:
        out_img = out_img.round()

    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.