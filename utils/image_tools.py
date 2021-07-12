import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_image_similarity(input_image, enhanced_image):
    L = input_image.max() - input_image.min()

    sim = ssim(input_image, enhanced_image, gaussian_weights=True,
               sigma=1.5, win_size=11, data_range=L)
    peak = psnr(input_image, enhanced_image, data_range=L)
    ambe = np.abs(np.mean(input_image)-np.mean(enhanced_image)) / L

    return sim, peak, ambe


def remap_image(enhanced_image, input_image):
    enhanced_image = np.squeeze(enhanced_image)
    input_image = np.squeeze(input_image)
    enh_min = enhanced_image.min()
    enh_max = enhanced_image.max()
    inp_min = input_image.min()
    inp_max = input_image.max()
    remapped_img = ((enhanced_image - enh_min) * (inp_max - inp_min))\
        / (enh_max - enh_min) \
        + inp_min
    return remapped_img
