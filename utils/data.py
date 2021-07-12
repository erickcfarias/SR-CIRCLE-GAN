import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import numpy as np
import logging
from glob import glob
import pandas as pd
import os
import re
from PIL import Image
from functools import reduce
from tifffile import imread, imsave
import subprocess
from utils.misc import format_i
from utils.image_tools import calculate_image_similarity
from tensorflow_addons.image import gaussian_filter2d


class DataLoader():

    def __init__(self, config):
        self.data_path = config['output_path']
        self.batch_size = int(config['batch_size'])
        self.noise_p = float(config['noise_prob'])
        self.noise_mu = float(config['noise_mu'])
        self.noise_sd = float(config['noise_sd'])
        self.blur_p = float(config['blur_prob'])

    def generate_dataset(self, data_type):

        gen = self.generator(data_type)

        dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.float32, tf.float32),
            (
                tf.TensorShape([self.batch_size, None, None, 1]),
                tf.TensorShape([self.batch_size, None, None, 1])
            )
        )

        df_iterator = iter(
            dataset
        )

        return df_iterator

    def generator(self, data_type):

        if data_type == 'train':
            files = glob(self.data_path + 'train/*')
            self.training = True

        elif data_type == 'test':
            files = glob(self.data_path + 'test/*')
            self.training = False

        random.shuffle(files)

        def gen():
            for img_file in files:

                lr, hr = self.read_image(img_file)

                lr_batch = tf.map_fn(
                    lambda x:
                    self.preprocess_image(x, augment=self.training),
                    elems=lr
                )

                hr_batch = tf.map_fn(
                    lambda x:
                    self.preprocess_image(x, augment=False),
                    elems=hr
                )

                yield lr_batch, hr_batch
        return gen

    def read_image(self, img_path):
        img = imread(img_path)
        lr_batch = tf.cast(img[0], tf.float32)
        hr_batch = tf.cast(img[1], tf.float32)

        return lr_batch, hr_batch

    @tf.function(experimental_relax_shapes=True)
    def preprocess_image(self, image, augment):

        image = tf.expand_dims(image, 2)

        if augment:
            if random.random() <= self.noise_p:
                sd = random.uniform(self.noise_mu, self.noise_sd)
                image = self.add_noise(image, mu=self.noise_mu, sd=sd)

            if random.random() <= self.blur_p:
                image = gaussian_filter2d(
                    image, filter_shape=(8, 8), sigma=1.
                )

        return image

    def add_noise(self, image, mu, sd):
        # Adding Gaussian noise
        noise = tf.cast(tf.random.normal(shape=tf.shape(image), mean=mu,
                                         stddev=sd, dtype=tf.double),
                        tf.float32)
        noise_img = tf.add(image, tf.abs(noise))
        return noise_img


class DeepLesionPreprocessor:

    def __init__(self, config: dict):
        self.logger = self._get_logger()
        self.config = config
        self.logger.info(
            'Preprocessor loaded.')
        self.urls = self._select_download_urls('training')

    def _select_download_urls(self, data_split):
        test_links = [
            'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
            'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
            'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
            'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
            'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
            'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
            'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
            'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
            'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
            'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip'
        ]
        training_links = [
            'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
            'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
            'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
            'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
            'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',
            'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
            'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
            'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
            'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
            'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
            'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
            'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
            'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
            'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',
            'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
            'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
            'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
            'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
            'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
            'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
            'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
            'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
            'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',
            'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
            'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
            'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
            'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
            'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
            'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
            'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
            'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
            'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',            
            'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
            'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
            'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
            'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
            'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
            'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
            'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
            'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
            'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',
            'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
            'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
            'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
            'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
            'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
        ]
        if data_split == 'training':
            urls = random.choices(training_links, k=5)
        else:
            urls = random.choices(test_links, k=3)

        return urls

    def _get_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        return logging.getLogger(__name__)

    def _check_dirs(self):
        if not os.path.isdir(self.config['input_path']):
            os.mkdir(self.config['input_path'])
            self.logger.info('Folder {} created.'.format(
                self.config['input_path']))

        if not os.path.isdir(self.config['output_path']):
            os.mkdir(self.config['output_path'])
            self.logger.info('Folder {} created.'.format(
                self.config['output_path']))

        if not os.path.isdir(self.config['output_path'] + 'train'):
            os.mkdir(self.config['output_path'] + 'train')
            self.logger.info(
                'Folder {}train created.'.format(self.config['output_path']))

        if not os.path.isdir(self.config['output_path'] + 'test'):
            os.mkdir(self.config['output_path'] + 'test')
            self.logger.info('Folder {}test created.'.format(self.config['output_path']))

        if not os.path.isdir(self.config['output_path'] + 'tune'):
            os.mkdir(self.config['output_path'] + 'tune')
            self.logger.info('Folder {}tune created.'.format(
                self.config['output_path']))

    def _download_data(self, idx, url):
        self.logger.info(
            'Started download file {}/{}.'.format(idx+1, len(self.urls)))

        if not os.path.isdir('download/'):
            os.mkdir('download/')

        # download file
        bashCommand = 'wget {} -O ./download/file.zip'.format(url)
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

        # unzip file
        bashCommand = 'unzip ./download/file.zip -d ./download/'
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

        # rm file
        bashCommand = 'rm ./download/file.zip'
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

        if self.config['arrange_folders']:
            # move and rename images
            self.logger.info(
                'Renaming images from file {}/{}.'.format(idx+1, len(self.urls)))
            images = glob('download/*/*/*')
            for img in images:
                try:
                    x = re.split(r'/', string=img)
                    new_path = '{}/{}_{}'.format(self.config['input_path'], x[2], x[3]).\
                        replace("//", "/")
                    os.rename(img, new_path)
                except Exception as e:
                    self.logger.exception(e)
                    continue

    def _prepare_training(self, part):
        """ for each patient:
                - list all image slices belonging to patient
                    for i in range(20):
                        - select a slice
                        - perform a random crop
                        - validate if patch has less than 50% of air
                        - save patch
        """
        self.logger.info('Started generating random patches from raw image files.')
        self.files = glob(self.config['input_path'] + '*')
        self.file_names = [re.search(pattern=r'\d.+\d.png', string=i)[0]
                           for i in self.files]
        self.file_idxs = list(
            set([
                re.sub(pattern=r'_\d{3}.png', repl='', string=f)
                for f in self.file_names
            ])
        )

        batches_generated = 0
        target_batches = int(self.config['epoch_size'] // self.config['batch_size'] // 5)

        while batches_generated < target_batches:

            # generate random bounding box sizes outside of the batch loop,
            # because batch must have same dims
            x_size = random.randint(
                self.config['input_size_min'], self.config['input_size_max']
            )
            while x_size % self.config['upscale_rate'] != 0:
                x_size = x_size + 1

            y_size = random.randint(
                self.config['input_size_min'], self.config['input_size_max']
            )
            while y_size % self.config['upscale_rate'] != 0:
                y_size = y_size + 1

            batch_size = 0
            batch_files_lr = []
            batch_files_hr = []
            while batch_size < self.config['batch_size']:

                # select a random inpatient and a random slice image from it
                idx = random.choice(self.file_idxs)
                files = [f for f in self.files if idx in f]
                rand_img = np.random.choice(files)

                # select a random coordinate for cropping
                x = random.randint(0, 512 - x_size)
                y = random.randint(0, 512 - y_size)

                img = sitk.ReadImage(rand_img)
                img = sitk.GetArrayFromImage(img)
                img = self.transform_to_hu(img)
                img = self.normalize_hu(img)
                img = tf.expand_dims(img, 2)
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)

                # perform the crop - HR
                hr = tf.image.crop_to_bounding_box(img, y, x, y_size, x_size)

                # perform the crop - LR
                lr = tf.image.resize(
                    img,
                    (int(img.shape[0] // self.config['upscale_rate']),
                     int(img.shape[1] // self.config['upscale_rate'])),
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=True
                )
                lr = tf.image.crop_to_bounding_box(
                    lr,
                    int(y // self.config['upscale_rate']),
                    int(x // self.config['upscale_rate']),
                    int(y_size // self.config['upscale_rate']),
                    int(x_size // self.config['upscale_rate'])
                )
                lr = tf.image.resize(
                    lr,
                    (int(hr.shape[0]), int(hr.shape[1])),
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=True
                )

                lr = np.squeeze(lr)
                lr = lr.astype(np.float16)
                hr = np.squeeze(hr)
                hr = hr.astype(np.float16)

                # validate if more than 50% of image is composed of interest tissue
                if self._validate_img_air_proportion(hr, 0.5):
                    batch_files_hr.append(np.expand_dims(hr, 0))
                    batch_files_lr.append(np.expand_dims(lr, 0))
                    batch_size += 1

            # save batch file
            # shape [resolution_type, img_n, x, y]
            lr_batch = reduce(lambda x, y: np.concatenate(
                [x, y], axis=0), batch_files_lr)

            hr_batch = reduce(lambda x, y: np.concatenate(
                [x, y], axis=0), batch_files_hr)        

            batch = np.concatenate([
                np.expand_dims(lr_batch, 0),
                np.expand_dims(hr_batch, 0)
            ], axis=0)

            imsave(
                self.config['output_path'] +
                'train/batch_' + str(part) + "_" + str(batches_generated+1) + '.tif',
                batch
            )
            if batches_generated % 50 == 0:
                self.logger.info("{} batches generated.".format(batches_generated))

            batches_generated += 1

    def transform_to_hu(self, image):
        image = image * 1. + (-32768.)
        return image

    def normalize_hu(self, image):
        # Our values currently range from -1024 to around 2000.
        # Anything above 300 is not interesting to us,
        # as these are simply bones with different radiodensity.
        # A commonly used set of thresholds in the LUNA16
        # competition to normalize between are -1000 and 400.
        if self.config['scaler'] == 'tanh':
            image = 2. * \
                ((image - self.config['hu_scale_min']) /
                 (self.config['hu_scale_max'] - self.config['hu_scale_min'])) - 1.
            image[image > 1.] = 1.
            image[image < -1.] = -1.
        elif self.config['scaler'] == 'sigm':
            image = (image - self.config['hu_scale_min']) / \
                (self.config['hu_scale_max'] - self.config['hu_scale_min'])
            image[image > 1.] = 1.
            image[image < 0.] = 0.
        return image

    def _prepare_testing(self):
        """ Select randomly X batches from training folder and move them to test
        """
        images = glob(self.config['output_path'] + 'train/*')
        images = np.random.choice(images, size=5, replace=False)

        for img in images:
            x = re.split(r'/', string=img)
            new_path = '{}{}'.format(
                self.config['output_path'] + "test/", x[2])
            os.rename(img, new_path)

    def _delete_folder(self, folder_path: str):
        bashCommand = "rm -r {}".format(folder_path)
        subprocess.run(
            bashCommand.split(),
            stdout=subprocess.PIPE,
            check=True
        )

    def _validate_img_air_proportion(self, img: np.array, proportion_threshold: float) -> bool:

        mask = img <= 0.05
        prop_air = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        if prop_air <= proportion_threshold:
            return True
        else:
            return False

    def run(self):
        for idx, url in enumerate(self.urls):
            self._check_dirs()
            if self.config['download']:
                self._download_data(idx, url)
                if self.config['delete_download']:
                    self._delete_folder('download/')
            if self.config['train']:
                self._prepare_training(idx)
                self.logger.info(
                    'Finished generating random patches for training.')
            if self.config['test']:
                self._prepare_testing()
                self.logger.info(
                    'Finished picking some batches for test.')
            if self.config['delete_raw']:
                self._delete_folder(self.config['input_path'])
