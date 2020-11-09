import atexit
import os
import os.path as osp
import shutil
import signal
import subprocess
import threading
import time
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

from lcnn.config import C, M
from lcnn.utils import recursive_to


class Trainer(object):
    def __init__(self, device, model, optimizer, train_loader, val_loader, out, dynamic_weight=None):
        self.device = device

        self.model = model
        self.optim = optimizer
        print('learning rate initialized as', self.optim.param_groups[0]["lr"])

        self.dynamic_weights=dynamic_weight
        if self.dynamic_weights:
            print('--------------------------')
            print('learnable weights are used')
            print('initialized as: ',self.dynamic_weights.params)
            print('--------------------------')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.run_tensorboard()
        time.sleep(1)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def run_tensorboard(self):
        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        p = subprocess.Popen(
            ["tensorboard", f"--logdir={board_out}", f"--port={C.io.tensorboard_port}"]
        )

        def killme():
            os.kill(p.pid, signal.SIGTERM)

        atexit.register(killme)

    def _loss(self, result):
        # [stack*dict{'center':torch.Tensor([1,6]),'corner':torch.Tensor([1,6]),'corner_offset':torch.Tensor([1,6]),'corner_bin_offset':torch.Tensor([1,6])}]
        losses = result["losses"]

        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys()) # [sum,center,corner,corner_offset,corner_bin_offset]
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)]) #[2,5]
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            stack_loss=[]
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                # printed losses are not multiplied with weight !!!
                self.metrics[i, 0] += self.dynamic_weights.params[j-1]*loss.item() # add loss to stack sum
                self.metrics[i, j] += self.dynamic_weights.params[j-1]*loss.item() # add loss to stack branch                
                stack_loss.append(loss)
            assert len(stack_loss) == 4, 'loss shape should be [1,4]'
            total_loss += self.dynamic_weights(stack_loss[0], stack_loss[1], stack_loss[2], stack_loss[3])
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * M.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * M.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, target,_) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }
                result = self.model(input_dict)

                total_loss += self._loss(result)

                # H = result["preds"]
                # for i in range(H["corner"].shape[0]):
                #     index = batch_idx * M.batch_size_eval + i
                #     np.savez(
                #         f"{npz}/{index:06}.npz",
                #         **{k: v[i].cpu().numpy() for k, v in H.items()},
                #     )
                #     if index >= 20:
                #         continue
                #     # self._plot_samples(i, index, H, target, f"{viz}/{index:06}")

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self.mean_loss = total_loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.module.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.module.state_dict(),
                "best_mean_loss": self.best_mean_loss,
                "dynamic": self.dynamic_weights.state_dict(),
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (image, target, _) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0 # []
            input_dict = {
                "image": recursive_to(image, self.device),
                "target": recursive_to(target, self.device),
                "mode": "training",
            }
            result = self.model(input_dict)

            loss = self._loss(result) # calculate total loss of this epoch
            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1 # ???
            self.iteration += 1 # count number of iteration
            self._write_metrics(1, loss.item(), "training", do_print=False) # Writes entries directly to event files in the log_dir to be consumed by TensorBoard.

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| " # epoch/amount of data that has been used
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            num_images = self.batch_size * self.iteration
            if num_images % self.validation_interval == 0 or num_images == 600: # validate every 600 image ot ...
                self.validate()
                time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )
        return total_loss

    def _plot_samples(self, i, index, result, target, prefix):
        fn = self.val_loader.dataset.filelist[index][:-10].replace("_a0", "") + ".png"
        img = io.imread(fn)
        imshow(img), plt.savefig(f"{prefix}_img.jpg"),plt.close()
        # object center heatmap

        mask_result = result["center"][i].cpu().numpy() # 1,128,128
        mask_target = target["center"][i].cpu().numpy() # 1,128,128
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia), plt.savefig(f"{prefix}_center_{ch}a.jpg"), plt.close()
            imshow(ib), plt.savefig(f"{prefix}_center_{ch}b.jpg"), plt.close() # b is gt

        # object center heatmap
        mask_result = result["corner"][i].cpu().numpy() # 1,128,128
        mask_target = target["corner"][i].cpu().numpy() # 1,128,128
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia), plt.savefig(f"{prefix}_corner_{ch}a.jpg"), plt.close()
            imshow(ib), plt.savefig(f"{prefix}_corner_{ch}b.jpg"), plt.close() # b is gt

        # object center heatmap
        mask_result = result["corner_offset"][i].cpu().numpy() # 1,2,128,128
        mask_target = target["corner_offset"][i].cpu().numpy() # 1,2,128,128
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia[0]), plt.savefig(f"{prefix}_corner_offset_x_{ch}a.jpg"), plt.close()
            imshow(ib[0]), plt.savefig(f"{prefix}_corner_offset_x_{ch}b.jpg"), plt.close() # b is gt
            imshow(ia[1]), plt.savefig(f"{prefix}_corner_offset_y_{ch}a.jpg"), plt.close()
            imshow(ib[1]), plt.savefig(f"{prefix}_corner_offset_y_{ch}b.jpg"), plt.close() # b is gt

        mask_result = result["corner_bin_offset"][i].cpu().numpy() # 1,2,128,128
        mask_target = target["corner_bin_offset"][i].cpu().numpy() # 1,2,128,128
        for ch, (ia, ib) in enumerate(zip(mask_target, mask_result)):
            imshow(ia[0]), plt.savefig(f"{prefix}_corner_bin_offset_x_{ch}a.jpg"), plt.close()
            imshow(ib[0]), plt.savefig(f"{prefix}_corner_bin_offset_x_{ch}b.jpg"), plt.close() # b is gt
            imshow(ia[1]), plt.savefig(f"{prefix}_corner_bin_offset_y_{ch}a.jpg"), plt.close()
            imshow(ib[1]), plt.savefig(f"{prefix}_corner_bin_offset_y_{ch}b.jpg"), plt.close() # b is gt

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch: # decrease lr by multiply 0.1 after certain epoches
                print('learning rate decreased')
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()
            print('--------------------------')
            print(f'weights in epoch {self.epoch} are {self.dynamic_weights.params}')
            print('--------------------------')


cmap = plt.get_cmap("jet")
# norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap) #, norm=norm
# sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im):
    plt.close()
    plt.tight_layout()
    plt.imshow(im)
    plt.colorbar(fraction=0.046) # sm
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def _launch_tensorboard(board_out, port, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])

    def kill():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(kill)

