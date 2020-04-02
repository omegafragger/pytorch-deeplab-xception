# Script to visualise outputs from neural networks

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

import scipy.misc as msc

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torchvision.utils import save_image

from colour_map import label_to_color_image


class Visualizer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        self.model = model

        self.evaluator = Evaluator(self.nclass)
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.evaluator = Evaluator(self.nclass)
        # Loading checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.train_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            #pred = output.data.cpu().numpy()
            #target = target.cpu().numpy()
            pred = torch.argmax(output, axis=1)

            entropy = self.softmax_entropy(output)
            
            # Code to visualize the prediction and target
            mask = ((target >= 0) & (target < self.nclass)).int()

            pred_im = (pred * mask).float()
            #pred_im = pred
            target_im = (target * mask).float()
            #target_im = target
            #entropy = (entropy * mask).float()
            #print(pred_im)
            #print (target_im)
            pred_im_numpy = pred_im[0].int().cpu().numpy()
            target_im_numpy = target_im[0].int().cpu().numpy()
            pred_im_color = label_to_color_image(pred_im_numpy)
            target_im_color = label_to_color_image(target_im_numpy)

            pred_imm = Image.fromarray((pred_im_color * 255).astype(np.uint8))
            target_imm = Image.fromarray((target_im_color * 255).astype(np.uint8))
            pred_imm.save('./predictions/pred_imm' + str(i) + '.jpg')
            target_imm.save('./targets/target_imm' + str(i) + '.jpg')

            save_image(image, './images/image' + str(i) + '.jpg')
            save_image(entropy, './entropies/entropy' + str(i) + '.jpg')
            #save_image(pred_im, 'pred_im' + str(i) + '.jpg')
            #save_image(target_im, 'target_im' + str(i) + '.jpg')


    def softmax_entropy(self, model_output):
        output = torch.squeeze(torch.nn.functional.softmax(model_output), 0)
        jitter = torch.ones(output.shape).to(output.device)
        jitter.fill_(1e-10)
        log_output = torch.log(output + jitter)
        prod = output * log_output
        prod = - torch.sum(prod, dim=0)
        return prod


    def save_stochastic_images(self, images, targets, outputs, pred_entropies, mutual_infos, counter=0):
        num_images = images.shape[0]
        count = counter
        for i in range(num_images):
            image = torch.unsqueeze(images[i], dim=0)
            target = torch.unsqueeze(targets[i], dim=0)
            output = torch.unsqueeze(outputs[i], dim=0)
            pred_entropy = pred_entropies[i]
            mutual_info =  mutual_infos[i]

            #print (pred_entropy)
            print (mutual_info)

            pred = torch.argmax(output, axis=1)
            mask = ((target >= 0) & (target < self.nclass)).int()

            pred_im = (pred * mask).float()
            target_im = (target * mask).float()

            pred_im_numpy = pred_im[0].int().cpu().numpy()
            target_im_numpy = target_im[0].int().cpu().numpy()
            pred_im_color = label_to_color_image(pred_im_numpy)
            target_im_color = label_to_color_image(target_im_numpy)

            pred_imm = Image.fromarray((pred_im_color * 255).astype(np.uint8))
            target_imm = Image.fromarray((target_im_color * 255).astype(np.uint8))
            pred_imm.save('./predictions/pred_imm' + str(count) + '.jpg')
            target_imm.save('./targets/target_imm' + str(count) + '.jpg')

            save_image(image, './images/image' + str(count) + '.jpg')
            save_image(pred_entropy, './pred_entropies/pred_entropy' + str(count) + '.jpg')
            save_image(mutual_info, './mutual_infos/mutual_info' + str(count) + '.jpg')
            count += 1
        return count


    def validation_mc_dropout(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')

        counter = 0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                model_outputs = self.get_stochastic_outputs(image)
                # Output is the mean of the stochastic model outputs
                output = torch.mean(model_outputs, 1)

            predictive_entropy = self.get_predictive_entropy(model_outputs)
            mutual_information = self.get_mutual_information(model_outputs)

            counter = self.save_stochastic_images(image, target, output, predictive_entropy, mutual_information, counter)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


    def get_stochastic_outputs(self, image):
        num_sfp = self.args.num_forward_passes
        # Model must be in train mode to get dropout layers working
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        with torch.no_grad():
            model_outputs = None
            for i in range(num_sfp):
                stochastic_output = self.model(image)
                if (i == 0):
                    model_outputs = torch.unsqueeze(stochastic_output, 1)
                else:
                    model_outputs = torch.cat((model_outputs, torch.unsqueeze(stochastic_output, 1)), 1)
        return model_outputs


    def get_predictive_entropy(self, stochastic_outputs):
        stochastic_outputs = torch.nn.functional.softmax(stochastic_outputs, dim=2)
        mean_output = torch.mean(stochastic_outputs, dim=1)
        jitter = torch.ones(mean_output.shape).to(mean_output.device)
        jitter.fill_(1e-10)
        log_mean_output = torch.log(mean_output + jitter)
        prod_output = mean_output * log_mean_output
        pred_entropy = - torch.sum(prod_output, dim=1)
        return pred_entropy


    def get_mutual_information(self, stochastic_outputs):
        pred_entropy = self.get_predictive_entropy(stochastic_outputs)
        stochastic_outputs = torch.nn.functional.softmax(stochastic_outputs, dim=2)
        jitter = torch.ones(stochastic_outputs.shape).to(stochastic_outputs.device)
        jitter.fill_(1e-10)
        log_stochastic_outputs = torch.log(stochastic_outputs + jitter)
        prod_stochastic_outputs = stochastic_outputs * log_stochastic_outputs
        prod_stochastic_outputs = torch.sum(torch.mean(prod_stochastic_outputs, dim=1), dim=1)
        mutual_information = pred_entropy + prod_stochastic_outputs
        return mutual_information



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'resnet_dropout', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')

    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    parser.add_argument('--num-forward-passes', type=int, default=10,
                        help='number of stochastic forward passes')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)


    visualizer = Visualizer(args)
    visualizer.validation_mc_dropout(1)

if __name__ == "__main__":
   main()