###################################
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import random
import math
import imutil
###################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset , DataLoader
###################################


def main():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./', train=False,
                                            download=True, transform=transform)


    my_openmixup_data = openmixup(trainset[0][0])
    imutil.show(my_openmixup_data, display=False,
                filename="./my_openmixup_data.jpg")

    my_opencutout = OpenCutout(trainset, 10)
    my_opencutmix = OpenCutmix(trainset, 10)


    my_opencutout_data, _ = my_opencutout.generate()
    my_opencutmix_data, _ = my_opencutmix.generate()

    imutil.show(my_opencutout_data, display=False,
                filename="./my_opencutout_data.jpg")

    imutil.show(my_opencutmix_data, display=False,
                filename="./my_opencutmix_data.jpg")

    print("Generated!")
    exit(0)

def openmixup(data):

    """
    Function:
        Given images, construct its OpenMixup-version images.

    Input:
        param1: tensor ( tensor for imgs)

    Usage:
        # It will take a few seconds.
        # openmixup(data).
    """

    mix_rate = 1 / 2

    data2 = data.clone()
    data2 = transforms.RandomRotation(360)(data2)

    data2 = data2

    data2 = mix_rate * data + (1 - mix_rate) * data2

    return data2


class OpenCutout(object):

    def __init__(self, trainset, batch_num):

        """
        Function:
            Given a dataset, construct its OpenCutout-version dataset.

        Input:
            param1: tuple (imgs, targets),   param2: int (how many batches you want to generate, referring to your training batches.)

        Usage:
            # Build OpenCutout dataset. It will take a few minutes.
            myOpenCutout = OpenCutout(trainset, batch_num)

            # Generate a batch of OpenCutout samples. It will take a few seconds.
            myOpenCutout.generate()

        """

        self.trainset = trainset
        self.auxiliary_positive_samples = []
        self.negative_samples = []
        self.original_labels = []
        self.trivial_masks = []

        self.negative_sample_num = 0
        self.batch_num = batch_num
        self.point = 0
        self.order = []

        self.grabcut()
        self.batchsize = int(self.negative_sample_num / self.batch_num)
        self.shuffle()



    def shuffle(self):
        self.order = torch.randperm(self.negative_sample_num)
        self.point = 0

    def grabcut(self):

        progress_bar = tqdm(self.trainset)
        for i, (images, labels) in enumerate(progress_bar):


            images = torch.unsqueeze(images, dim=0)
            labels = torch.tensor(labels)
            labels = torch.unsqueeze(labels, dim=0)

            # Get images, their foreground masks, and original labels using grabcut algorithm.
            # Bit 1 stands for foreground.
            selected_pic, fg_mask, label = grabcut(images, labels, 0.1, 0.8)

            if selected_pic is None:
                continue

            trivial_mask = fg_mask.clone()

            # Leverage fg_mask to generate some trivial masks.
            trivial_mask = torchvision.transforms.Resize(int(selected_pic.shape[2]/2))(trivial_mask)
            trivial_mask = (trivial_mask!=0)*1
            buff = torch.zeros(fg_mask.shape)
            buff[:,:,int(selected_pic.shape[2]/4):int(selected_pic.shape[2]/4*3),int(selected_pic.shape[2]/4):int(selected_pic.shape[2]/4*3)]=trivial_mask
            trivial_mask = buff


            # Initialize
            self.auxiliary_positive_samples.append(selected_pic)
            self.negative_samples.append(fg_mask)
            self.original_labels.append(label)
            self.trivial_masks.append(trivial_mask)


        self.negative_sample_num = len(self.auxiliary_positive_samples)
        print("finish grabcut!")


    # Get a batch of data
    def get_data(self):

        if self.point+self.batchsize > self.negative_sample_num:
            self.shuffle()

        selected_index = [self.order[k].item() for k in range(self.point,self.point+self.batchsize)]


        selected_pic = torch.cat([self.auxiliary_positive_samples[k] for k in selected_index],dim = 0)
        selected_fg = torch.cat([self.negative_samples[k] for k in selected_index], dim=0)
        selected_labels = torch.cat([self.original_labels[k] for k in selected_index], dim=0)

        self.point = self.point + self.batchsize

        return selected_pic, selected_fg, selected_labels


    # Generate trivial masks
    def get_trivial_mask(self):

        # random order
        shuffle_index = torch.randperm(self.negative_sample_num)
        # random location
        random_x = random.randint(0,self.trivial_masks[0].shape[1])
        random_y = random.randint(0, self.trivial_masks[0].shape[1])

        selected_index = [shuffle_index[k].item() for k in range(self.batchsize)]
        tensor = torch.cat([self.trivial_masks[k] for k in selected_index], dim=0)
        buff = torch.zeros([tensor.shape[0], tensor.shape[1], tensor.shape[2] * 2, tensor.shape[3] * 2])
        buff[:, :, int(buff.shape[2] / 4):int(buff.shape[2] / 4 * 3),int(buff.shape[3] / 4):int(buff.shape[3] / 4 * 3)] = tensor
        return_mask = buff[:,:,random_x:random_x+int(buff.shape[2]/2),random_y:random_y+int(buff.shape[2]/2)]

        return return_mask


    # Generate a batch of samples including negative samples and auxiliary_positive_samples
    # to prevent shortcuts.
    def generate(self):

        selected_pic, selected_fg, selected_labels = self.get_data()
        # Obtain trivial masks to generate auxiliary_positive_samples
        trivial_masks = self.get_trivial_mask()[:selected_pic.shape[0]]

        negative_samples = selected_pic * ((selected_fg == 0) * 1)
        auxiliary_positive_samples = selected_pic * ((trivial_masks == 0) * 1)

        # no overlap between foregrounds and trivial masks
        selected_index = torch.sum((selected_fg * ((auxiliary_positive_samples == 0) * 1)), dim=[1, 2, 3]) == 0
        auxiliary_positive_samples = auxiliary_positive_samples[selected_index]

        # Negative labels
        negative_lables = selected_labels.clone()
        nclass = 6 # Assume there are six known classes. Assign an independent label id.
        negative_lables[:] = nclass+1
        # Maintaining labels
        auxiliary_positive_labels = selected_labels.clone()
        auxiliary_positive_labels = auxiliary_positive_labels[selected_index]
        # Optional label switch. Depend on main.py
        # auxiliary_positive_labels = torch.tensor([splits.index(int(auxiliary_positive_labels[k])) for k in range(len(auxiliary_positive_labels))])

        # Concatenate
        data = torch.cat([negative_samples, auxiliary_positive_samples], dim=0)
        targets = torch.cat([negative_lables, auxiliary_positive_labels], dim=0)

        return data, targets

class OpenCutmix(object):

    def __init__(self, trainset, batch_num):

        """
        Function:
            Given a dataset, construct its OpenCutmix-version dataset.

        Input:
            param1: tuple (imgs, targets),   param2: int (how many batches you want to generate, referring to your training batches.)

        Usage:
            # Build OpenCutmix dataset. It will take a few minutes.
            myOpenCutmix = OpenCutmix(trainset, batch_num)

            # Generate a batch of OpenCutmix samples. It will take a few seconds.
            myOpenCutmix.generate()

        """

        self.trainset = trainset
        self.negative_samples = []
        self.auxiliary_positive_samples = []
        self.original_labels = []
        self.trivial_masks = []

        self.negative_sample_num = 0
        self.batch_num = batch_num
        self.point = 0
        self.order = []

        self.grabcut()
        self.batchsize = int(self.negative_sample_num / self.batch_num)
        self.shuffle()


    def shuffle(self):
        self.order = torch.randperm(self.negative_sample_num)
        self.point = 0

    def grabcut(self):

        progress_bar = tqdm(self.trainset)
        for i, (images, labels) in enumerate(progress_bar):

            images = torch.unsqueeze(images, dim=0)
            labels = torch.tensor(labels)
            labels = torch.unsqueeze(labels, dim=0)

            # Get images, their foreground masks, and original labels using grabcut algorithm.
            # Bit 1 stands for foreground.
            selected_pic, fg_mask, label = grabcut(images, labels, 0.1, 0.8)

            if selected_pic is None:
                continue

            trivial_mask = fg_mask.clone()

            # Leverage fg_mask to generate some trivial masks.
            trivial_mask = torchvision.transforms.Resize(int(selected_pic.shape[2] / 2))(trivial_mask)
            trivial_mask = (trivial_mask != 0) * 1
            buff = torch.zeros(fg_mask.shape)
            buff[:, :, int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3),
            int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3)] = trivial_mask
            trivial_mask = buff

            # Initialize
            self.auxiliary_positive_samples.append(selected_pic)
            self.negative_samples.append(fg_mask)
            self.original_labels.append(label)
            self.trivial_masks.append(trivial_mask)

        self.negative_sample_num = len(self.auxiliary_positive_samples)
        print("finish grabcut!")

    # Get a batch of data
    def get_data(self):

        if self.point + self.batchsize > self.negative_sample_num:
            self.shuffle()

        selected_index = [self.order[k].item() for k in range(self.point, self.point + self.batchsize)]

        selected_pic = torch.cat([self.auxiliary_positive_samples[k] for k in selected_index], dim=0)
        selected_fg = torch.cat([self.negative_samples[k] for k in selected_index], dim=0)
        selected_labels = torch.cat([self.original_labels[k] for k in selected_index], dim=0)

        self.point = self.point + self.batchsize

        return selected_pic, selected_fg, selected_labels

    # Generate trivial masks
    def get_trivial_mask(self):

        # random order
        shuffle_index = torch.randperm(self.negative_sample_num)
        # random location
        random_x = random.randint(0, self.trivial_masks[0].shape[1])
        random_y = random.randint(0, self.trivial_masks[0].shape[1])

        selected_index = [shuffle_index[k].item() for k in range(self.batchsize)]
        tensor = torch.cat([self.trivial_masks[k] for k in selected_index], dim=0)
        buff = torch.zeros([tensor.shape[0], tensor.shape[1], tensor.shape[2] * 2, tensor.shape[3] * 2])
        buff[:, :, int(buff.shape[2] / 4):int(buff.shape[2] / 4 * 3),
        int(buff.shape[3] / 4):int(buff.shape[3] / 4 * 3)] = tensor
        return_mask = buff[:, :, random_x:random_x + int(buff.shape[2] / 2),
                      random_y:random_y + int(buff.shape[2] / 2)]

        return return_mask

    # Generate a batch of samples including negative samples and auxiliary_positive_samples
    # to prevent shortcuts.
    def generate(self):

        selected_pic, selected_fg, selected_labels = self.get_data()
        masks = self.get_trivial_mask()[:selected_pic.shape[0]]
        # Get filler for replacing
        filler = flip_for_openset(selected_pic)

        negative_samples = selected_pic * ((selected_fg == 0) * 1)*0.5 + filler * ((selected_fg == 1) * 1)*0.5
        # No overlap between foregrounds and trivial masks
        masks_judge = selected_pic * ((masks == 0) * 1)
        auxiliary_positive_samples = selected_pic * ((masks == 0) * 1)*0.5 + filler * ((masks == 1) * 1)*0.5
        selected_index = torch.sum((selected_fg * ((masks_judge == 0) * 1)), dim=[1, 2, 3]) == 0
        auxiliary_positive_samples = auxiliary_positive_samples[selected_index]

        # Negative labels
        negative_sample_labels = selected_labels.clone()
        nclass = 6  # Assume there are six known classes. Assign an independent label id.
        negative_sample_labels[:] = nclass+2
        # Maintaining labels
        auxiliary_positive_sample_labels = selected_labels.clone()
        auxiliary_positive_sample_labels = auxiliary_positive_sample_labels[selected_index]
        # Optional label switch. Depend on main.py
        # for index_ in range(len(auxiliary_positive_sample_labels)):
        #     auxiliary_positive_sample_labels[index_] = splits.index(int(auxiliary_positive_sample_labels[index_]))

        data = torch.cat([negative_samples, auxiliary_positive_samples], dim=0)
        targets = torch.cat([negative_sample_labels, auxiliary_positive_sample_labels], dim=0)

        return data, targets

def grabcut(images, targets, lowerbound, upperbound):

    """
        Function:
            Given images, output their foreground masks using grabcut algorithm.

        Input:
            param3: float (0<param3<1, filter out tiny foreground masks),   param4: float (0<param4<1, filter out large foreground masks)

    """



    selected_pic = None
    foreground = None
    labels = None

    for k in range(len(images)):

        # try:

        image = images[k]

        image_test = transforms.ToPILImage()(image)

        img = cv2.cvtColor(np.asarray(image_test), cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)

        img = np.array(img)

        height, width, _ = img.shape

        x1 = 1
        x2 = width - 2
        y1 = 1
        y2 = height - 2

        rect = (x1, y1, x2, y2)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[:] = cv2.GC_PR_FGD
        mask[0, :] = cv2.GC_BGD
        mask[height - 1, :] = cv2.GC_BGD
        mask[:, 0] = cv2.GC_BGD
        mask[:, width - 1] = cv2.GC_BGD

        # Dummy placeholders
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        # Iteratively extract foreground object from background
        cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

        # Remove background from image
        fg_mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        fg_mask = torchvision.transforms.ToTensor()(fg_mask)



        # filter out irregular foregrounds
        if (torch.sum(fg_mask) > height * width * lowerbound) and (
                torch.sum(fg_mask) < height * width * upperbound):

            tensor_buff = torch.unsqueeze(image, dim=0)
            if selected_pic == None:
                selected_pic = tensor_buff
            else:
                selected_pic = torch.cat([selected_pic, tensor_buff], dim=0)

            tensor_buff = torch.unsqueeze(fg_mask, dim=0)
            if foreground == None:
                foreground = tensor_buff
            else:
                foreground = torch.cat([foreground, tensor_buff], dim=0)

            tensor_buff = torch.unsqueeze(targets[k], dim=0)
            if labels == None:
                labels = tensor_buff
            else:
                labels = torch.cat([labels, tensor_buff], dim=0)


    return selected_pic, foreground, labels

def flip_for_openset(data):


    image_h, image_w = data.shape[2:]

    data = torchvision.transforms.RandomHorizontalFlip()(data)
    data = torchvision.transforms.RandomVerticalFlip()(data)

    data2 = data.clone()

    data_buff = torchvision.transforms.functional.vflip(data2)

    data2[:,:,:int(image_h/2),:] = data[:,:,int(image_h/2):,:]
    data2[:,:,int(image_h/2):,:] = data_buff[:,:,:int(image_h/2),:]

    data_buff = torchvision.transforms.functional.hflip(data2)

    data2[:,:,:,int(image_h/2):] = data2[:,:,:,0:int(image_h/2)]
    data2[:, :, :, 0:int(image_h / 2)] = data_buff[:,:,:,int(image_h/2):]

    data2 = transforms.RandomRotation(360)(data2)


    return data2




if __name__ == '__main__':
    main()