from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder

class MUNIT(object) :
    def __init__(self, sess, args):
        self.model_name = 'MUNIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch
        self.iteration = args.iteration

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.num_style = args.num_style # for test

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.gan_w = args.gan_w
        self.recon_x_w = args.recon_x_w
        self.recon_s_w = args.recon_s_w
        self.recon_c_w = args.recon_c_w
        self.recon_x_cyc_w = args.recon_x_cyc_w

        """ Generator """
        self.n_res = args.n_res
        self.mlp_dim = args.mlp_dim

        self.n_downsample = args.n_sample
        self.n_upsample = args.n_sample
        self.style_dim = args.style_dim

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.all_dataset = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# dataset : ", self.all_dataset)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# style in test phase : ", self.num_style)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def Style_Encoder(self, x, reuse=False, scope='style_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(2) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='down_conv_'+str(i))

            x = adaptive_avg_pooling(x)
            x = conv(x, self.style_dim, kernel=1, stride=1, scope='SE_logit')

            return x

    def Content_Encoder(self, x, reuse=False, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            for i in range(self.n_downsample) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = instance_norm(x, scope='ins_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            return x

    def generator(self, contents, style, reuse=False, scope="decoder"):
        channel = self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            mu, sigma = self.MLP(style, reuse)
            x = None
            for i in range(self.n_res) :
                x = adaptive_resblock(contents, channel, mu, sigma, scope='adaptive_resblock'+str(i))

            for i in range(self.n_upsample) :
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_'+str(i))
                x = layer_norm(x, scope='layer_norm_'+str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    def MLP(self, x, reuse=False, scope='MLP'):
        channel = self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            x = linear(x, channel, scope='linear_0')
            x = relu(x)

            x = linear(x, channel, scope='linear_1')
            x = relu(x)

            mu = linear(x, channel, scope='mu')
            sigma = linear(x, channel, scope='sigma')

            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel])

            return mu, sigma

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        for scale in range(self.n_scale) :

            if scale > 0 : reuse = True

            with tf.variable_scope(scope, reuse=reuse) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_0')

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                    x = lrelu(x, 0.2)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, scope='D_logit')
                D_logit.append(x)

                x_init = down_sample(x_init)

        return D_logit

    def Encoder_A(self, x_A, reuse=False):
        style_A = self.Style_Encoder(x_A, reuse=reuse, scope='style_encoder_A')
        content_A = self.Content_Encoder(x_A, reuse=reuse, scope='content_encoder_A')

        return content_A, style_A

    def Encoder_B(self, x_B, reuse=False):
        style_B = self.Style_Encoder(x_B, reuse=reuse, scope='style_encoder_B')
        content_B = self.Content_Encoder(x_B, reuse=reuse, scope='content_encoder_B')

        return content_B, style_B

    def Decoder_A(self, content_B, style_A, reuse=False):
        x_ba = self.generator(contents=content_B, style=style_A, reuse=reuse, scope='decoder_A')

        return x_ba

    def Decoder_B(self, content_A, style_B, reuse=False):
        x_ab = self.generator(contents=content_A, style=style_B, reuse=reuse, scope='decoder_B')

        return x_ab


    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        trainA = trainA.prefetch(self.batch_size).shuffle(self.all_dataset).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        trainB = trainB.prefetch(self.batch_size).shuffle(self.all_dataset).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()

        trainA_iterator = trainA.make_one_shot_iterator()

        trainB_iterator = trainB.make_one_shot_iterator()


        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()


        """ Define Encoder, Generator, Discriminator """
        self.style_a = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_a')
        self.style_b = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_b')

        # encode
        content_a, style_a_prime = self.Encoder_A(self.domain_A)
        content_b, style_b_prime = self.Encoder_B(self.domain_B)

        # decode (within domain)
        x_aa = self.Decoder_A(content_B=content_a, style_A=style_a_prime)
        x_bb = self.Decoder_B(content_A=content_b, style_B=style_b_prime)

        # decode (cross domain)
        x_ba = self.Decoder_A(content_B=content_b, style_A=self.style_a, reuse=True)
        x_ab = self.Decoder_B(content_A=content_a, style_B=self.style_b, reuse=True)

        # encode again
        content_b_, style_a_ = self.Encoder_A(x_ba, reuse=True)
        content_a_, style_b_ = self.Encoder_B(x_ab, reuse=True)

        # decode again (if needed)
        if self.recon_x_cyc_w > 0 :
            x_aba = self.Decoder_A(content_B=content_a_, style_A=style_a_prime, reuse=True)
            x_bab = self.Decoder_B(content_A=content_b_, style_B=style_b_prime, reuse=True)

            cyc_recon_A = L1_loss(x_aba, self.domain_A)
            cyc_recon_B = L1_loss(x_bab, self.domain_B)

        else :
            cyc_recon_A = 0.0
            cyc_recon_B = 0.0


        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        G_ad_loss_a = generator_loss(fake_A_logit)
        G_ad_loss_b = generator_loss(fake_B_logit)

        D_ad_loss_a = discriminator_loss(real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(real_B_logit, fake_B_logit)

        recon_A = L1_loss(x_aa, self.domain_A) # reconstruction
        recon_B = L1_loss(x_bb, self.domain_B) # reconstruction

        recon_style_A = L1_loss(style_a_, self.style_a)
        recon_style_B = L1_loss(style_b_, self.style_b)

        recon_content_A = L1_loss(content_a_, content_a)
        recon_content_B = L1_loss(content_b_, content_b)


        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.recon_x_w * recon_A + \
                           self.recon_s_w * recon_style_A + \
                           self.recon_c_w * recon_content_A + \
                           self.recon_x_cyc_w * cyc_recon_A

        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.recon_x_w * recon_B + \
                           self.recon_s_w * recon_style_B + \
                           self.recon_c_w * recon_content_B + \
                           self.recon_x_cyc_w * cyc_recon_B

        Discriminator_A_loss = self.gan_w * D_ad_loss_a
        Discriminator_B_loss = self.gan_w * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])


        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_image')
        self.test_style = tf.placeholder(tf.float32, [1, 1, 1, self.style_dim], name='test_style')

        test_content_a, _ = self.Encoder_A(self.test_image, reuse=True)
        test_content_b, _ = self.Encoder_B(self.test_image, reuse=True)

        self.test_fake_A = self.Decoder_A(content_B=test_content_b, style_A=self.test_style, reuse=True)
        self.test_fake_B = self.Decoder_B(content_A=test_content_a, style_B=self.test_style, reuse=True)


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if epoch > 0 :
                lr = lr / 2

            for idx in range(start_batch_id, self.iteration):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                train_feed_dict = {
                    self.style_a : style_a,
                    self.style_b : style_b,
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    # save_images(batch_B_images, [self.batch_size, 1],
                    #             './{}/real_B_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))

                    # save_images(fake_A, [self.batch_size, 1],
                    #             './{}/fake_A_{}_{:02d}_{:06d}.jpg'.format(self.sample_dir, gpu_id, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_image : sample_image, self.test_style : test_style})
                save_images(fake_img, [1, 1], image_path)
                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style):
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image, self.test_style: test_style})
                save_images(fake_img, [1, 1], image_path)
                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_size, self.img_size))
                index.write("</tr>")
        index.close()