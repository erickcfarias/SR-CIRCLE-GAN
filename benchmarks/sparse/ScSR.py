import numpy as np
from skimage.transform import resize
from scipy.signal import convolve2d
from tqdm import tqdm
from scipy import sparse


def fss_yang(lmbd, A, b):
    """
    L1QP_FeatureSign solves nonnegative quadradic programming
    using Feature Sign.
    min  0.5*x'*A*x+b'*x+\lambda*|x|
    [net,control]=NNQP_FeatureSign(net,A,b,control)
    """

    EPS = 1e-9
    x = np.zeros((A.shape[1], 1))
    # print('X =', x.shape)
    grad = np.dot(A, x) + b
    # print('GRAD =', grad.shape)
    ma = np.amax(np.multiply(abs(grad), np.isin(x, 0).T), axis=0)
    mi = np.zeros(grad.shape[1])
    for j in range(grad.shape[1]):
        for i in range(grad.shape[0]):
            if grad[i, j] == ma[j]:
                mi[j] = i
                break
    mi = mi.astype(int)
    # print(grad[mi])
    while True:

        if np.all(grad[mi]) > lmbd + EPS:
            x[mi] = (lmbd - grad[mi]) / A[mi, mi]
        elif np.all(grad[mi]) < - lmbd - EPS:
            x[mi] = (- lmbd - grad[mi]) / A[mi, mi]
        else:
            if np.all(x == 0):
                break

        while True:

            a = np.where(x != 0)
            Aa = A[a, a]
            ba = b[a]
            xa = x[a]

            vect = -lmbd * np.sign(xa) - ba
            x_new = np.linalg.lstsq(Aa, vect)
            idx = np.where(x_new != 0)
            o_new = np.dot((vect[idx] / 2 + ba[idx]).T,
                           x_new[idx]) + lmbd * np.sum(abs(x_new[idx]))

            s = np.where(np.multiply(xa, x_new) < 0)
            if np.all(s == 0):
                x[a] = x_new
                break

            x_min = x_new
            o_min = o_new
            d = x_new - xa
            t = np.divide(d, xa)
            for zd in s.T:
                x_s = xa - d / t[zd]
                x_s[zd] = 0
                idx = np.where(x_s == 0)
                o_s = np.dot((np.dot(Aa[idx, idx], x_s[idx]) / 2 +
                              ba[idx]).T, x_s[idx] + lmbd * np.sum(abs(x_s[idx])))
                if o_s < o_min:
                    x_min = x_s
                    o_min = o_s

            x[a] = x_min

        grad = np.dot(A, sparse.csc_matrix(x)) + b

        ma, mi = max(np.multiply(abs(grad), np.where(x == 0)))
        if ma <= lmbd + EPS:
            break

    return x


def extract_lr_feat(img_lr):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [[-1, 0, 1], ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = [[1, 0, -2, 0, 1], ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat


def create_list_step(start, stop, step):
    list_step = []
    for i in range(start, stop, step):
        list_step = np.append(list_step, i)
    return list_step


def lin_scale(xh, us_norm):
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))

    if hr_norm > 0:
        s = us_norm * 1.2 / hr_norm
        xh = np.multiply(xh, s)
    return xh


def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap):

    patch_size = 3

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_hr)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1,
                                       patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1,
                                       patch_size - overlap), img_us_height - patch_size - 1)

    count = 0

    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy: yy + patch_size, xx: xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy: yy +
                                       patch_size, xx: xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            b = np.dot(np.multiply(Dl.T, -1), y)
            w = fss_yang(lmbd, Dl, b)

            hr_patch = np.dot(Dh, w)
            hr_patch = lin_scale(hr_patch, us_norm)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy: yy + patch_size, xx: xx + patch_size] += hr_patch
            cnt_matrix[yy: yy + patch_size, xx: xx + patch_size] += 1

    index = np.where(cnt_matrix < 1)[0]
    img_hr[index] = img_us[index]

    cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)

    return img_hr
