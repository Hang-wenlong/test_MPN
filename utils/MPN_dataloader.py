"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import glob
from PIL import Image
import imageio


class tiles_splidate(Dataset):
    def __init__(self, X_Y_frame, transform=None):
        self.X_Y_frame = X_Y_frame 
        self.transform = transform

    def __len__(self):
        return  self.X_Y_frame.shape[0]

    def __getitem__(self, index):
        fn,label,sample_id = self.X_Y_frame.iloc[index,:]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label,sample_id ###
    
# image_data_set= tiles_splidate(Train_frame, data_trans)

# dataloaders =  DataLoader(image_data_set, 
#                           batch_size=32, 
#                           shuffle=True)


class MPNBags(data_utils.Dataset):
    def __init__(self, datasets_all, seed=1, train=True):
        self.datasets_all = datasets_all
        self.train = train
        self.r = np.random.RandomState(seed)

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _generate_batch(self, path):
        bags_list = []
        labels_list = []
        for each_path in path:
            each_path = each_path.replace('\\','/') ## 替换为D:\\图片\\Zbtv1.jpg
            # name_img = []
            img = []
            img_path = glob.glob(each_path + '/*.png') ##取出每bag的所有实例
            num_ins = len(img_path)
            # print(num_ins)
            # print(each_path)
    
            label = int(each_path.split('/')[-2]) ##取出每个bag的标签
    
            if label == 0:
                curr_label = np.zeros(num_ins, dtype=np.uint8) ##若bag为0，每个实例都是0
                
            elif label == 1:
                curr_label = np.ones(num_ins,dtype=np.uint8)  ##若bag为1，每个实例都是1
                
            elif label == 2:
                curr_label = 2 * np.ones(num_ins,dtype=np.uint8)  ##若bag为1，每个实例都是1
                
            elif label == 3:
                curr_label = 3 * np.ones(num_ins,dtype=np.uint8)  ##若bag为1，每个实例都是1
                
            for each_img in img_path:
                each_img = each_img.replace('\\','/') ## 替换为D:\\图片\\Zbtv1.jpg
                img_data = np.asarray(imageio.imread(each_img), dtype=np.float32)
                ##### 对每个实例进行归一化
                #img_data -= 255
                img_data[:, :, 0] -= 123.68
                img_data[:, :, 1] -= 116.779
                img_data[:, :, 2] -= 103.939
                img_data /= 255
                # sci.imshow(img_data)
                img.append(np.expand_dims(img_data,0))##第一维度加1
                # name_img.append(each_img.split('/')[-1])  ##每个实例的名字
            stack_img = torch.tensor(np.concatenate(img, axis=0))  ##所有实例封装一个300，256，256，3
            bags_list.append(stack_img)
            labels_list.append(torch.LongTensor(curr_label))
        return bags_list, labels_list
    def _create_bags(self):
        if self.train:
            bags_list, labels_list = self._generate_batch(self.datasets_all['test'])
        else:
            bags_list, labels_list = self._generate_batch(self.datasets_all['test'])

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
