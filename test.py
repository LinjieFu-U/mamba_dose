import copy
from absl import app, flags
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from diffusion import GaussianDiffusionSampler
from model import MambaUnet
from Dataset.dataset import Valid_Data, Test_Data
from Dataset.datasets import MyDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec
from PIL import Image

FLAGS = flags.FLAGS
import SimpleITK as sitk

# UNet
flags.DEFINE_integer("ch", 64, help="base channel of UNet")
flags.DEFINE_multi_integer("ch_mult", [1, 2, 2, 4, 4], help="channel multiplier")
flags.DEFINE_multi_integer("attn", [1], help="add attention to these levels")
flags.DEFINE_integer("num_res_blocks", 2, help="# resblock in each level")
flags.DEFINE_float("dropout", 0.0, help="dropout rate of resblock")

# Gaussian Diffusion
flags.DEFINE_float("beta_1", 1e-4, help="start beta value")
flags.DEFINE_float("beta_T", 0.02, help="end beta value")
flags.DEFINE_integer("T", 1000, help="total diffusion steps")
flags.DEFINE_enum(
    "mean_type", "epsilon", ["xprev", "xstart", "epsilon"], help="predict variable"
)
flags.DEFINE_enum(
    "var_type", "fixedlarge", ["fixedlarge", "fixedsmall"], help="variance type"
)

# Training
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("img_size", 128, help="image size")
flags.DEFINE_integer("num_workers", 0, help="workers of Dataloader")

# Logging & Sampling
flags.DEFINE_string("DIREC", "ddpm-unet", help="name of your project")
flags.DEFINE_integer("sample_size", 1, "sampling size of images")

device = torch.device("cuda:0")


def test():
    # dataset
    va_train = MyDataset("test")
    validloader = DataLoader(
        va_train,
        batch_size=FLAGS.sample_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # model setup
    net_model = MambaUnet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout,
    )
    ema_model = copy.deepcopy(net_model)

    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type,
    ).to(device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    checkpoint = torch.load(
        "./Save/" + FLAGS.DIREC + "/" + "tciadose-unet/" + "/model_latest.pkl"
    )
    net_model.load_state_dict(checkpoint["net_model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    restore_epoch = checkpoint["epoch"]
    print("Finish loading model")

    output = np.zeros(
        (9118, 6, 128, 128)
    )  # example size, please change based on your data
    lr = np.zeros((9118, 256, 256))
    hr = np.zeros((9118, 256, 256))
    if not os.path.exists("Output/" + FLAGS.DIREC):
        os.makedirs("Output/" + FLAGS.DIREC)
    net_model.eval()
    count = 1
    with torch.no_grad():
        # with tqdm(validloader, unit="batch") as tepoch:
        for data, target, mask, cbct_path, ct_path in tqdm(validloader):
            condition = data.to(device)
            length = data.shape[0]
            x_T = torch.randn(length, 1, FLAGS.img_size, FLAGS.img_size)
            min_val = x_T.min()
            max_val = x_T.max()
            x_T = (x_T - min_val) / (max_val - min_val)
            x_T = x_T.to(device)
            x_0 = ema_sampler(x_T, condition)
            fig = plt.figure()
            fig.set_figheight(8)
            fig.set_figwidth(28)
            spec = gridspec.GridSpec(
                ncols=7,
                nrows=2,
                width_ratios=[1, 1, 1, 1, 1, 1, 1],
                wspace=0.01,
                hspace=0.01,
                height_ratios=[1, 1],
                left=0,
                right=1,
                top=1,
                bottom=0,
            )
            img = data[0][0].data.squeeze()
            ax = fig.add_subplot(spec[0])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            count = 1
            for kk in range(len(x_0)):  # x_0 [10,b,1,h,w]
                imgs = x_0[kk].data.squeeze().cpu()  # imgs [b,h,w]
                imgs = imgs.numpy()
                # if(kk==4):
                #     imgs=imgs*(3071+1000)-1000
                #     itk_image = sitk.GetImageFromArray(imgs)
                #     sitk.WriteImage(itk_image, os.path.join(r'D:\test\syncbct',str(cbct_path[0].split('.')[0])+'.nii.gz'))

                # 估计图像的最小值和最大值
                min_val = np.min(imgs)
                max_val = np.max(imgs)

                # 将数据从 [min_val, max_val] 缩放到 [0, 255]
                scaled_img = (imgs - min_val) / (max_val - min_val) * 255

                # 将图像转换为 uint8 类型
                uint8_img = np.clip(scaled_img, 0, 255).astype(np.uint8)

                # 将 NumPy 数组转换为 Pillow 图像对象
                pil_img = Image.fromarray(uint8_img)

                # 保存图像到磁盘上
                if not os.path.exists(
                    "./Test_Output/" + FLAGS.DIREC + "/tciadose-unet-all-image"
                ):
                    os.makedirs(
                        "./Test_Output/" + FLAGS.DIREC + "/tciadose-unet-all-image"
                    )
                if not os.path.exists(
                    "./Test_Output/" + FLAGS.DIREC + "/tciadose-unet-all"
                ):
                    os.makedirs("./Test_Output/" + FLAGS.DIREC + "/tciadose-unet-all")

                if kk == 4:
                    try:
                        # image=sitk.ReadImage(cbct_path[0])
                        #
                        # uint_img = sitk.GetImageFromArray(scaled_img)
                        # uint_img.SetSpacing(image.GetSpacing())
                        # uint_img.SetOrigin(image.GetOrigin())
                        # uint_img.SetDirection(image.GetDirection())
                        # sitk.WriteImage(uint_img,
                        #                 './Test_Output/' + FLAGS.DIREC + '/' + 'syl-all-image' + '/' + str(
                        #                     ct_path[0]))
                        pil_img.save(
                            "./Test_Output/"
                            + FLAGS.DIREC
                            + "/"
                            + "tciadose-unet-all-image/"
                            + str(ct_path[0]).split("\\")[-4]
                            + "_"
                            + str(ct_path[0]).split("\\")[-1].split(".")[0]
                            + ".png"
                        )
                    except RuntimeError:
                        continue

                ax = fig.add_subplot(spec[count])
                ax.imshow(imgs, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                count += 1
                output[count : count + length, kk, :, :] = imgs
            img = target[0].data.squeeze().cpu()
            ax = fig.add_subplot(spec[6])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            plt.savefig(
                "./Test_Output/"
                + FLAGS.DIREC
                + "/"
                + "tciadose-unet-all/"
                + str(ct_path[0]).split("\\")[-4]
                + "_"
                + str(ct_path[0]).split("\\")[-1].split(".")[0]
                + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
            # lr[count:count+length,:,:] = data.squeeze().cpu()
            # hr[count:count+length,:,:] = target.squeeze().cpu()

    #         count += length
    #
    # path = 'Output/' + FLAGS.DIREC + '/result_epoch_' + str(restore_epoch) + '.hdf5'
    # f = h5py.File(path, 'w')
    # f.create_dataset('out', data=output)
    # f.create_dataset('lr', data=lr)
    # f.create_dataset('hr', data=hr)
    # f.close()


def main(argv):
    test()


if __name__ == "__main__":
    app.run(main)
