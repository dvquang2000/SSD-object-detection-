import cv2
import matplotlib.pyplot as plt

from utils.augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, \
    Expand, RandomSampleCrop, RandomMirror, ToPercentCoords,\
    Resize, SubtractMeans

from make_datapath import make_data_path_list
from lib import  *
from extract_inform_annotation import Anno_xml
class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), #convert image from int to float32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # change color by random
                Expand(color_mean),
                RandomSampleCrop(), #random crop image
                RandomMirror(), # xoay ảnh ngược lại
                ToPercentCoords(), #convert annotation data về dạng quy chuẩn 0->1
                Resize(input_size),
                SubtractMeans(color_mean) #trừ đi mean của BGR
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase] (img, boxes, labels)

if __name__ == "__main__":
    # Prepare train, valid, annottion list
    root_path = "./data/VOCdevkit/VOC2012"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)
    # read image
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path) # return Height, width, channel(BGR)
    height, width, channels = img.shape
    #annotation information
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    annotation_file_path = train_annotation_list[0]
    trans_anno = Anno_xml(classes)
    anno_infor_list = trans_anno(annotation_file_path, width, height)
    print(anno_infor_list)
    # plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # mặc định của matplotlib là RGB
    plt.show()

    #prepare data transform
    color_mean = (104, 117,123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # transform train img
    phase = "train"
    img_transformed, boxes, labels = transform(img,phase, anno_infor_list[:,:4], anno_infor_list[:,4])
    # plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(boxes)
    print(labels)

    #transform val img
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, anno_infor_list[:, :4], anno_infor_list[:, 4])
    # plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    # plt.show()
    print(boxes)
    print(labels)



