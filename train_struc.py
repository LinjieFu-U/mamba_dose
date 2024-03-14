import copy
import os
import warnings
import scipy.io as sio
from absl import app, flags
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from diffusion_struc import GaussianDiffusionTrainer, GaussianDiffusionSampler
# from model import MambaUnet
from Dataset.dataset import Train_Data, Valid_Data
from Dataset.datasets import MyDataset
from vim.mamba_struc_unet import MambaUnet



FLAGS = flags.FLAGS
flags.DEFINE_bool("train", True, help="train from scratch")
flags.DEFINE_bool("continue_train", True, help="train from scratch")

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
flags.DEFINE_float("lr", 1e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("img_size", 128, help="image size")
flags.DEFINE_integer("batch_size", 26, help="batch size")
flags.DEFINE_integer("num_workers", 8, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# Logging & Sampling
flags.DEFINE_string("DIREC", "ddpm-unet", help="name of your project")
flags.DEFINE_integer("sample_size", 1, "sampling size of images")
flags.DEFINE_integer(
    "max_epoch",
    1500,
    help="frequency of saving checkpoints, 0 to disable during training",
)

device = torch.device("cuda:0")


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def train():
    # dataset
    tr_train = MyDataset("train")
    trainloader = DataLoader(
        tr_train,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    va_train = MyDataset("val")
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

    init_lr = FLAGS.lr
    lr = FLAGS.lr

    optim = torch.optim.Adam(net_model.parameters(), lr=lr)

    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T
    ).to(device)
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

    if FLAGS.continue_train:
        checkpoint = torch.load(
            "./Save/"
            + FLAGS.DIREC
            + "/hxdose-struc-mamba-unet"
            + "/model_latest.pkl",
            map_location="cuda:0",
        )
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        restore_epoch = checkpoint["epoch"]
        print("Finish loading model")
    else:
        restore_epoch = 0

    if not os.path.exists("Loss"):
        os.makedirs("Loss")

    tr_ls = []
    if FLAGS.continue_train:
        readmat = sio.loadmat("./Loss/" + FLAGS.DIREC)
        load_tr_ls = readmat["loss"]
        for i in range(restore_epoch):
            tr_ls.append(load_tr_ls[0][i])
        print("Finish loading loss!")

    for epoch in range(restore_epoch, FLAGS.max_epoch):
        with tqdm(trainloader, unit="batch") as tepoch:
            tmp_tr_loss = 0
            tr_sample = 0
            net_model.train()
            for data, target,mask,cbct_path,ct_path in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                # train
                optim.zero_grad()
                condition = data.to(device)
                x_0 = target.to(device)
                x_0 = x_0.unsqueeze(1)
                loss = trainer(x_0, condition)
                tmp_tr_loss += loss.item()
                tr_sample += len(data)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                tepoch.set_postfix({"Loss": loss.item()})

        tr_ls.append(tmp_tr_loss / tr_sample)
        sio.savemat("./Loss/" + FLAGS.DIREC + ".mat", {"loss": tr_ls})

        if not os.path.exists("Train_Output/" + FLAGS.DIREC):
            os.makedirs("Train_Output/" + FLAGS.DIREC)
        net_model.eval()
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                for batch_idx, (data, target,mask,cbct_path,ct_path) in enumerate(
                    validloader
                ):
                    if batch_idx == 4:
                        x_T = torch.randn(
                            FLAGS.sample_size, 1, FLAGS.img_size, FLAGS.img_size
                        )
                        min_val = x_T.min()
                        max_val = x_T.max()
                        x_T = (x_T - min_val) / (max_val - min_val)
                        x_T = x_T.to(device)
                        condition = data.to(device)
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

                        img = data[0][0]
                        img = img.data.squeeze().cpu()

                        # 估计图像的最小值和最大值
                        ax = fig.add_subplot(spec[0])
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                        ax.axis("off")
                        # img = data[1].data.squeeze()
                        # ax = fig.add_subplot(spec[7])
                        # ax.imshow(img, cmap='gray', vmin=0,vmax=1)
                        # ax.axis('off')

                        count = 1
                        for kk in range(5):  # x_0 [5,b,1,h,w]
                            imgs = x_0[kk]  # imgs [b,1,h,w]
                            img = imgs[0].data.squeeze().cpu()

                            ax = fig.add_subplot(spec[count])
                            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                            ax.axis("off")

                            # imgs = x_0[kk] # imgs [b,1,h,w]
                            # img = imgs[1].data.squeeze().cpu()
                            # ax = fig.add_subplot(spec[count+7])
                            # ax.imshow(img, cmap='gray', vmin=0,vmax=1)
                            # ax.axis('off')

                            count += 1

                        img = target[0].data.squeeze().cpu()
                        ax = fig.add_subplot(spec[6])
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                        ax.axis("off")
                        if not os.path.exists(
                            "./Train_Output/"
                            + FLAGS.DIREC
                            + "/hxdose-struc-mamba-unet"
                        ):
                            os.makedirs(
                                "./Train_Output/"
                                + FLAGS.DIREC
                                + "/hxdose-struc-mamba-unet"
                            )
                        plt.savefig(
                            "./Train_Output/"
                            + FLAGS.DIREC
                            + "/hxdose-struc-mamba-unet"
                            + "/Epoch_"
                            + str(epoch + 1)
                            + ".png",
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close()

        # save
        if not os.path.exists("Save/" + FLAGS.DIREC):
            os.makedirs("Save/" + FLAGS.DIREC)
        ckpt = {
            "net_model": net_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch + 1,
            # 'x_T': x_T,
        }
        if not os.path.exists("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet"):
            os.makedirs("./Save/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet")
        if (epoch + 1) % 100 == 0:
            torch.save(
                ckpt,
                "./Save/"
                + FLAGS.DIREC
                + "/hxdose-struc-mamba-unet"
                + "/model_epoch_"
                + str(epoch + 1)
                + ".pkl",
            )
        torch.save(
            ckpt,
            "./Save/"
            + FLAGS.DIREC
            + "/hxdose-struc-mamba-unet"
            + "/model_latest.pkl",
        )


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
        "./Save/"
        + FLAGS.DIREC
        + "/"
        + "hxdose-struc-mamba-unet/"
        + "/model_latest.pkl"
    )
    net_model.load_state_dict(checkpoint["net_model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    restore_epoch = checkpoint["epoch"]
    print("Finish loading model")

    # output = np.zeros((9118, 6, 128, 128))  # example size, please change based on your data
    # lr = np.zeros((9118, 256, 256))
    # hr = np.zeros((9118, 256, 256))
    if not os.path.exists("Output/" + FLAGS.DIREC):
        os.makedirs("Output/" + FLAGS.DIREC)
    net_model.eval()
    count = 1
    with torch.no_grad():
        # with tqdm(validloader, unit="batch") as tepoch:
        for data, target,mask,cbct_path,ct_path in tqdm(validloader):
            condition = data.to(device)
            length = data.shape[0]
            x_T = torch.randn(length, 1, FLAGS.img_size, FLAGS.img_size)
            min_val = x_T.min()
            max_val = x_T.max()
            x_T = (x_T - min_val) / (max_val - min_val)
            x_T = x_T.to(device)
            import time

            # start=time.time()
            x_0 = ema_sampler(x_T, condition)
            # end=time.time()
            # elapsed_time = end - start
            # print(f"模型运行时间：{elapsed_time} 秒")
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
                    "./Test_Output/"
                    + FLAGS.DIREC
                    + "/hxdose-struc-mamba-unet-all-image"
                ):
                    os.makedirs(
                        "./Test_Output/"
                        + FLAGS.DIREC
                        + "/hxdose-struc-mamba-unet-all-image"
                    )
                if not os.path.exists(
                    "./Test_Output/" + FLAGS.DIREC + "/hxdose-struc-mamba-unet-all"
                ):
                    os.makedirs(
                        "./Test_Output/"
                        + FLAGS.DIREC
                        + "/hxdose-struc-mamba-unet-all"
                    )

                if kk == 4:
                    try:
                        pil_img.save(
                            "./Test_Output/"
                            + FLAGS.DIREC
                            + "/"
                            + "hxdose-struc-mamba-unet-all-image/"
                            + str(ct_path[0]).split("/")[-3]
                            + "_"
                            + str(ct_path[0]).split("/")[-1].split(".")[0]
                            + ".png"
                        )
                    except RuntimeError:
                        continue

                ax = fig.add_subplot(spec[count])
                ax.imshow(imgs, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                count += 1
                # output[count:count + length, kk, :, :] = imgs
            img = target[0].data.squeeze().cpu()
            ax = fig.add_subplot(spec[6])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            plt.savefig(
                "./Test_Output/"
                + FLAGS.DIREC
                + "/"
                + "hxdose-struc-mamba-unet-all/"
                + str(ct_path[0]).split("/")[-3]
                + "_"
                + str(ct_path[0]).split("/")[-1].split(".")[0]
                + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


def train_main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action="ignore", category=FutureWarning)
    if FLAGS.train:
        train()
    test_main(argv)


def test_main(argv):
    test()


if __name__ == "__main__":
    app.run(train_main)
