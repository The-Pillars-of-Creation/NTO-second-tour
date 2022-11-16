import torch
from torch.utils.data import Dataset 
import pandas as pd
import torchvision
import numpy as np
import PIL
import collections 
import random
import cv2
import os
import pytorchvideo.models.resnet
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning


class VideoLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        video = self.dataframe.iloc[index].path
        label = self.dataframe.iloc[index].labels 
        if self.transform:
            video = self.transform(video)
        return video, label


class VideoFilePathToTensor(object):
    def __init__(self, max_len=None, fps=None, padding_mode=None):
        self.max_len = max_len
        self.fps = fps
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode
        self.channels = 3  # only available to read 3 channels video

    def __call__(self, path):
        # open video file
        cap = cv2.VideoCapture(path)
        assert(cap.isOpened())

        # calculate sample_factor to reset fps
        sample_factor = 1
        if self.fps:
            old_fps = cap.get(cv2.CAP_PROP_FPS)  # fps of video
            sample_factor = int(old_fps / self.fps)
            assert(sample_factor >= 1)
        
        # init empty output frames (C x L x H x W)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(int(num_frames / sample_factor), self.max_len)
        else:
            # time length is unlimited
            time_len = int(num_frames / sample_factor)

        frames = torch.FloatTensor(self.channels, time_len, height, width)

        for index in range(time_len):
            frame_index = sample_factor * index

            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1)
                frames[:, index, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[:, index:, :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert(index > 0)
                    frames[:, index:, :, :] = frames[:, index-1, :, :].view(self.channels, 1, height, width)
                break

        frames /= 255
        cap.release()
        return frames


class VideoResize(object):
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, collections.Iterable) and len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        h, w = self.size
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)
        
        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


dataset = VideoLabelDataset(
	"train.csv",
    transform=torchvision.transforms.Compose([
        VideoFilePathToTensor(max_len=50, fps=10, padding_mode='last'),
        VideoResize([256, 256]),
    ])
)

train_set, test_set = torch.utils.data.random_split(dataset, [450, 72])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 8, shuffle = True)