from model.models import discriminator, generator, sft_generator, sa_generator
from utils.cloud import S3Manager
from utils.image_tools import calculate_image_similarity
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, Progbar
import subprocess
import os
import pickle as pkl


class ganCIRCLE:

    def __init__(self, data_loader, config, s3_key=None, s3_secret=None):

        self.config = config
        self.lambda_0 = config['validation_loss_weight']
        self.lambda_1 = config['cycle_loss_weight']
        self.lambda_2 = config['identity_loss_weight']
        self.lambda_3 = config['joint_loss_weight']
        self.lambda_4 = config['supervision_weight']
        self.checkpoint_every = config['checkpoint_every']
        self.version_name = config['version_name']
        self.s3_key = s3_key
        self.s3_secret = s3_secret

        # Initialize dataset iterators
        self.data_loader = data_loader
        self.batch_size = config['batch_size']

        # Create Folders for checkpointing and images saving
        self._create_folders()

        # Build Models

        # Build the critics
        self.d_hr = discriminator(img_shape=(None, None, 1),
                                  spectr_norm=config['spectral_normalization'])
        self.d_lr = discriminator(img_shape=(None, None, 1),
                                  spectr_norm=config['spectral_normalization'])

        # Build the generators
        if config['conditioning'] == 'sft':
            self.g_hr_lr = sft_generator(
                img_shape=(None, None, 1),
                hu_min=config['hu_scale_min'],
                hu_max=config['hu_scale_max'],
                spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )
            self.g_lr_hr = sft_generator(
                img_shape=(None, None, 1),
                hu_min=config['hu_scale_min'],
                hu_max=config['hu_scale_max'],
                spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )

        elif config['conditioning'] == 'self-attention':
            self.g_hr_lr = sa_generator(
                img_shape=(None, None, 1), spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )
            self.g_lr_hr = sa_generator(
                img_shape=(None, None, 1), spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )

        else:
            self.g_hr_lr = generator(
                img_shape=(None, None, 1), spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )
            self.g_lr_hr = generator(
                img_shape=(None, None, 1), spectr_norm=config['spectral_normalization'],
                gen_out=config['generator_output_activation']
            )

        # Later, whenever we perform an optimization step, we pass in the step.
        self.D_LR = config['learning_rate']
        self.G_LR = config['learning_rate'] * config['ttur_rate']

        # Build Optimizers
        self.d_lr_optim = Adam(self.D_LR, beta_1=0.5, beta_2=0.9)
        self.d_hr_optim = Adam(self.D_LR, beta_1=0.5, beta_2=0.9)
        self.g_a_optim = Adam(self.G_LR, beta_1=0.5, beta_2=0.9)
        self.g_b_optim = Adam(self.G_LR, beta_1=0.5, beta_2=0.9)

        # Initilize Checkpointer and Restore state
        self._build_checkpointer()
        # self.plot_models()

    def _create_folders(self):

        # Create the RUN folder for checkpointing
        self.run_folder = './runs/{}/'.format(self.version_name)
        try:
            # Clear existing folders
            bashCommand = 'rm {} -r'.format(self.run_folder)
            subprocess.run(
                bashCommand.split(),
                stdout=subprocess.PIPE,
                check=True
            )
        except Exception:
            pass

        if not os.path.exists("./runs/"):
            os.mkdir("./runs/")

        if not os.path.exists(self.run_folder):
            os.mkdir(self.run_folder)

        if not os.path.exists(os.path.join(self.run_folder, 'checkpoints')):
            os.mkdir(os.path.join(self.run_folder, 'checkpoints'))

        if not os.path.exists(
                os.path.join(self.run_folder, 'checkpoints/tensorboard_log')
        ):
            os.mkdir(os.path.join(self.run_folder, 'checkpoints/tensorboard_log'))

    def _build_checkpointer(self):
        # Build Tensorboard Summary Writer
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.run_folder, 'checkpoints/tensorboard_log')
        )
        self.summary_writer.set_as_default()

        # Instantiate S3 Manager
        if (self.s3_key is not None) & (self.s3_secret is not None):
            print('Starting S3 checkpointer.')
            self.cloud_manager = S3Manager(
                key=self.s3_key,
                secret=self.s3_secret
            )

            # Load checkpoints from S3
            self.cloud_manager.download_all_files(
                bucket='thesis-checkpoint',
                version_name=self.version_name,
                save_dir=os.path.join(self.run_folder, 'checkpoints')
            )

        # Build Checkpointer
        self.checkpointer = tf.train.Checkpoint(
            step=tf.Variable(1),
            g_a_optimizer=self.g_a_optim,
            g_b_optimizer=self.g_b_optim,
            g_lr_hr=self.g_lr_hr,
            g_hr_lr=self.g_hr_lr,
            d_lr_optim=self.d_lr_optim,
            d_lr=self.d_lr,
            d_hr_optim=self.d_hr_optim,
            d_hr=self.d_hr
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpointer,
            directory=os.path.join(self.run_folder, 'checkpoints'),
            max_to_keep=1
        )

        # Restore Checkpoint
        self.checkpointer.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:

            # Restore Params
            with open(os.path.join(
                      self.run_folder, 'checkpoints/params.pkl'
                      ), 'rb') as f:
                data = pkl.load(f)
                self.D_LR = data[0]
                self.G_LR = data[1]
                self.lambda_0 = data[2]
                self.lambda_1 = data[3]
                self.lambda_2 = data[4]
                self.lambda_3 = data[5]

            print("Restored from {}".format(
                self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    @tf.function(experimental_relax_shapes=True)
    def gradient_penalty(self, discriminator, real_images, fake_images):

        epsilon = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_image = epsilon * \
            real_images + (1 - epsilon) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_image)
            d_interpolated = discriminator(interpolated_image, training=True)

        gradients = gp_tape.gradient(d_interpolated, [interpolated_image])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @tf.function(experimental_relax_shapes=True)
    def disc_train_step(self, imgs_lr, imgs_hr):

        with tf.GradientTape(persistent=True) as d_tape:

            # fake images
            fake_hr_img = self.g_lr_hr(imgs_lr, training=True)
            fake_lr_img = self.g_hr_lr(imgs_hr, training=True)

            # Discriminator Loss
            real_hr_validity = self.d_hr(imgs_hr, training=True)
            fake_hr_validity = self.d_hr(fake_hr_img, training=True)
            real_lr_validity = self.d_lr(imgs_lr, training=True)
            fake_lr_validity = self.d_lr(fake_lr_img, training=True)

            # D_HR Wasserstein Loss
            d_hr_loss =\
                tf.reduce_mean(fake_hr_validity)\
                - tf.reduce_mean(real_hr_validity)

            # D_LR Wasserstein Loss
            d_lr_loss =\
                tf.reduce_mean(fake_lr_validity)\
                - tf.reduce_mean(real_lr_validity)

            # Gradient Penalty - D_HR
            d_hr_gp = self.gradient_penalty(self.d_hr, imgs_hr, fake_hr_img)

            # Gradient Penalty - D_LR
            d_lr_gp = self.gradient_penalty(self.d_lr, imgs_lr, fake_lr_img)

            # Total Discriminator Loss
            d_hr_loss = d_hr_loss + 10*d_hr_gp
            d_lr_loss = d_lr_loss + 10*d_lr_gp

            d_total_loss = (d_hr_loss + d_lr_loss) * 0.5


        # Backpropagate D loss
        d_hr_grad, d_lr_grad = d_tape.gradient(
            d_total_loss,
            (
                self.d_hr.trainable_variables,
                self.d_lr.trainable_variables
            )
        )
        self.d_hr_optim.apply_gradients(
            zip(d_hr_grad, self.d_hr.trainable_variables))

        self.d_lr_optim.apply_gradients(
            zip(d_lr_grad, self.d_lr.trainable_variables))

        return d_total_loss

    @tf.function(experimental_relax_shapes=True)
    def gen_train_step(self, imgs_lr, imgs_hr):

        with tf.GradientTape(persistent=True) as g_tape:

            # Translate images to the other domain
            # G(X), F(Y)
            fake_hr = self.g_lr_hr(imgs_lr, training=True)
            fake_lr = self.g_hr_lr(imgs_hr, training=True)

            # Translate images back to original domain
            # F(G(X)), G(F(Y))
            reconstr_lr = self.g_hr_lr(fake_hr, training=True)
            reconstr_hr = self.g_lr_hr(fake_lr, training=True)

            # Identity mapping of images
            # G(Y), F(X)
            img_hr_id = self.g_lr_hr(imgs_hr, training=True)
            img_lr_id = self.g_hr_lr(imgs_lr, training=True)

            # Discriminators determines validity of translated images
            valid_lr = self.d_lr(fake_lr, training=True)
            valid_hr = self.d_hr(fake_hr, training=True)

            # Adversarial Loss
            val_loss_a = -tf.reduce_mean(valid_lr)
            val_loss_b = -tf.reduce_mean(valid_hr)

            # Cycle L1 Loss
            cyc_loss_a = tf.keras.losses.MeanAbsoluteError()(
                imgs_lr, reconstr_lr
            )
            cyc_loss_b = tf.keras.losses.MeanAbsoluteError()(
                imgs_hr, reconstr_hr
            )

            # Identity L1 Loss
            id_loss_a = tf.keras.losses.MeanAbsoluteError()(imgs_hr, img_hr_id)
            id_loss_b = tf.keras.losses.MeanAbsoluteError()(imgs_lr, img_lr_id)

            # Joint Loss
            x_var_a = fake_hr[:, :, 1:, :] - fake_hr[:, :, :-1, :]
            y_var_a = fake_hr[:, 1:, :, :] - fake_hr[:, :-1, :, :]
            joint_loss_a = tf.reduce_sum(
                tf.abs(x_var_a)) + tf.reduce_sum(tf.abs(y_var_a))

            img_diff = imgs_hr - fake_hr
            x_var_b = img_diff[:, :, 1:, :] - img_diff[:, :, :-1, :]
            y_var_b = img_diff[:, 1:, :, :] - img_diff[:, :-1, :, :]
            joint_loss_b = tf.reduce_sum(
                tf.abs(x_var_b)) + tf.reduce_sum(tf.abs(y_var_b))

            joint_loss = (
                tf.constant(0.1) * joint_loss_a
                + tf.constant(.9) * joint_loss_b
            )

            # Supervision Loss
            sup_loss_a = tf.keras.losses.MeanAbsoluteError()(imgs_hr, fake_hr)
            sup_loss_b = tf.keras.losses.MeanAbsoluteError()(imgs_lr, fake_lr)
            sup_loss = sup_loss_a + sup_loss_b

            # Total GAN Loss
            g_total_loss = \
                self.lambda_0 * (val_loss_a + val_loss_b) \
                + self.lambda_1 * (cyc_loss_a + cyc_loss_b) \
                + self.lambda_2 * (id_loss_a + id_loss_b) \
                + self.lambda_3 * joint_loss \
                + self.lambda_4 * sup_loss

        # Backpropagate Generators' losses
        g_lr_hr_grad, g_hr_lr_grad = g_tape.gradient(
            g_total_loss,
            (
                self.g_lr_hr.trainable_variables,
                self.g_hr_lr.trainable_variables
            )
        )

        self.g_a_optim.apply_gradients(
            zip(g_lr_hr_grad, self.g_lr_hr.trainable_variables))

        self.g_b_optim.apply_gradients(
            zip(g_hr_lr_grad, self.g_hr_lr.trainable_variables))

        return g_total_loss

    def train(self, epochs):
        iterations = tf.constant(
            (self.config['epoch_size']
             // (self.config['batch_size'])) - 1
        )
        tf.print('Each epoch will have %s iterations' % (iterations.numpy()))

        # Single image to follow-up during training
        self.test_it = self.data_loader.generate_dataset(data_type='test')
        test_batch = self.test_it.get_next()

        pb = Progbar(tf.cast(iterations * epochs, tf.float32),
                     width=10, verbose=1)
        for epoch in tf.range(epochs):
            g_total_loss_list = []
            d_total_loss_list = []
            self.train_it = self.data_loader.generate_dataset(data_type='train')

            tf.print("\n epoch {}/{}".format(epoch, epochs))

            for i in tf.range(iterations):

                try:
                    train_batch = self.train_it.get_next()

                    lr_batch = train_batch[0]
                    hr_batch = train_batch[1]

                    d_total_loss = self.disc_train_step(
                        lr_batch, hr_batch
                    )
                    d_total_loss_list.append(d_total_loss)

                    g_total_loss = self.gen_train_step(
                        lr_batch, hr_batch
                    )
                    g_total_loss_list.append(g_total_loss)

                    pb.add(1.)

                except tf.python.framework.errors_impl.OutOfRangeError:
                    # tf.print("\n OutOfRangeError - Regenerating tf.Datasets")
                    self.train_it = self.data_loader.generate_dataset(data_type='train')
                    self.test_it = self.data_loader.generate_dataset(data_type='test')

                except tf.python.framework.errors_impl.InvalidArgumentError:
                    # tf.print("\n %s" % (e))
                    self.train_it = self.data_loader.generate_dataset(data_type='train')
                    self.test_it = self.data_loader.generate_dataset(data_type='test')
                    continue

            # Log metrics on tensorboard
            test_lr = test_batch[0]
            test_hr = test_batch[1]

            test_sr = self.g_lr_hr(test_lr)

            # Log sample images into tensorboard
            log_images = tf.concat([
                tf.expand_dims(test_lr[0], 0),
                tf.expand_dims(test_lr[1], 0),
                tf.expand_dims(test_hr[0], 0),
                tf.expand_dims(test_hr[1], 0),
                tf.expand_dims(test_sr[0], 0),
                tf.expand_dims(test_sr[1], 0),
            ], axis=0)
            tf.summary.image(
                "Image log", log_images, max_outputs=6,
                step=int(self.checkpointer.step)
            )

            # Calculate SSIM, PSNR and AMBE for all test_batch
            ssim_list = []
            psnr_list = []
            ambe_list = []
            for lr, hr, sr in zip(test_lr, test_hr, test_sr):
                ssim, psnr, ambe = calculate_image_similarity(
                    np.squeeze(sr),
                    np.squeeze(hr)
                )

                ssim2, psnr2, ambe2 = calculate_image_similarity(
                    np.squeeze(lr),
                    np.squeeze(hr)
                )

                ssim_list.append(ssim/ssim2)
                psnr_list.append(psnr/psnr2)
                ambe_list.append(ambe/ambe2)

            tf.summary.scalar(
                'ssim', data=np.mean(ssim_list),
                step=int(self.checkpointer.step)
            )
            tf.summary.scalar(
                'psnr', data=np.mean(psnr_list),
                step=int(self.checkpointer.step)
            )
            tf.summary.scalar(
                'ambe', data=np.mean(ambe_list),
                step=int(self.checkpointer.step)
            )
            tf.summary.scalar(
                'gen_loss', data=tf.reduce_mean(g_total_loss_list[-30:]),
                step=int(self.checkpointer.step)
            )
            tf.summary.scalar(
                'disc_loss', data=tf.reduce_mean(d_total_loss_list[-30:]),
                step=int(self.checkpointer.step)
            )

            # Checkpoint
            self.tf_checkpoint()

    @tf.function
    def tf_checkpoint(self):
        tf.py_function(self.checkpoint, inp=[], Tout=[])

    def checkpoint(self):
        self.checkpointer.step.assign_add(1)
        if int(self.checkpointer.step) % self.checkpoint_every == 0:
            self.ckpt_manager.save()
            with open(
                    os.path.join(self.run_folder, 'checkpoints', 'params.pkl'),
                    'wb') as f:
                pkl.dump([
                    self.D_LR,  self.G_LR,
                    self.lambda_0, self.lambda_1,
                    self.lambda_2, self.lambda_3
                ], f)

        if (self.s3_key is not None) & (self.s3_secret is not None):
            self.cloud_manager.clear_folder(
                bucket='thesis-checkpoint',
                prefix=self.version_name + '/'
            )
            for file in os.listdir(os.path.join(self.run_folder,
                                                'checkpoints')):

                if not os.path.isdir(file):
                    self.cloud_manager.\
                        upload_file(bucket='thesis-checkpoint',
                                    file_name=os.path.join(
                                        self.run_folder, 'checkpoints', file
                                    ),
                                    object_name=self.version_name + '/' + file)

            for file in os.listdir(os.path.join(self.run_folder,
                                                'checkpoints/tensorboard_log')):

                if not os.path.isdir(file):

                    self.cloud_manager\
                        .upload_file(
                            bucket='thesis-checkpoint',
                            file_name=os.path.join(
                                    self.run_folder, "checkpoints/tensorboard_log", file
                                ),
                            object_name=self.version_name + "/tensorboard_log/" + file
                        )

    def plot_models(self):
        plot_model(self.d_lr, to_file=os.path.join(
            self.run_folder, 'plot_model/d_lr.png'), show_shapes=True,
            show_layer_names=plot_model, expand_nested=True)
        plot_model(self.d_hr, to_file=os.path.join(
            self.run_folder, 'plot_model/d_hr.png'), show_shapes=True,
            show_layer_names=plot_model, expand_nested=True)
        plot_model(self.g_hr_lr, to_file=os.path.join(
            self.run_folder, 'plot_model/g_hr_lr.png'), show_shapes=True,
            show_layer_names=plot_model, expand_nested=True)
        plot_model(self.g_lr_hr, to_file=os.path.join(
            self.run_folder, 'plot_model/g_lr_hr.png'), show_shapes=True,
            show_layer_names=plot_model, expand_nested=True)
