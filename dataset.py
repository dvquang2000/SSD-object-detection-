import numpy as np
import torch

from lib import *

from make_datapath import make_data_path_list

from transform import DataTransform

from extract_inform_annotation import Anno_xml


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img, gt, height, width = self.pull_item(item)
        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape
        #get anno information

        anno_file_path = self.anno_list[index]
        anno_info = self.anno_xml(anno_file_path, width, height)

        #preprocessing

        img, boxes, labels = self.transform(img, self.phase, anno_info[:,:4], anno_info[:, 4])

        # BGR -> RGB, heght, width, channels -> channels, height, width
        img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        #groundtruth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def my_collate_fn(batch):
    target = []
    imgs = []
    for sammple in batch:
        imgs.append(sammple[0]) # sample[0] = img
        target.append(torch.FloatTensor(sammple[1])) #sample[1] = anno

    imgs = torch.stack(imgs, dim =0)
    return imgs, target




if __name__ == "__main__":
    root_path = "./data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase ="train",
                              transform=DataTransform(input_size,color_mean), anno_xml=Anno_xml(classes))
    # print(train_dataset.__getitem__(1))

    val_dataset = MyDataset(val_img_list, val_annotation_list, phase="val",
                              transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
    batch_size = 4

    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter)
    print(images.size())
    print(len(targets))
    print(targets[3].size()) #xmin, ymin, xmax, ymax, labels



