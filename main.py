from MUNIT import MUNIT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of MUNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='summer2winter', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--num_style', type=int, default=3, help='number of styles to sample')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
    parser.add_argument('--recon_x_w', type=float, default=10.0, help='weight of image reconstruction loss')
    parser.add_argument('--recon_s_w', type=float, default=1.0, help='weight of style reconstruction loss')
    parser.add_argument('--recon_c_w', type=float, default=1.0, help='weight of content reconstruction loss')
    parser.add_argument('--recon_x_cyc_w', type=float, default=0.0, help='weight of explicit style augmented cycle consistency loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--mlp_dim', type=int, default=256, help='number of filters in MLP')
    parser.add_argument('--style_dim', type=int, default=8, help='length of style code')
    parser.add_argument('--n_sample', type=int, default=2, help='number of sampling layers in content encoder')
    parser.add_argument('--n_res', type=int, default=4, help='number of residual blocks in content encoder/decoder')

    parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
    parser.add_argument('--n_scale', type=int, default=3, help='number of scales')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = MUNIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()