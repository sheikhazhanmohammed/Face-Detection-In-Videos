#importing libraries
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import os
from tqdm import tqdm
import torch
from sklearn.cluster import DBSCAN

#function to detect faces in a given image
#takes an input image and returns list of cropped and aligned faces from the image
def face_extractor_from_frame(img):
    faces = mtcnn(img)
    if faces is None:
        return None
    return faces

#function to save an image to the disk
def write_image_to_disk(cropped_faces, save_path_root):
    for i, c in enumerate(cropped_faces):
        c = c.numpy().transpose(1,2,0)
        c = (c+1)/2
        c = c*255.0
        c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        savepath = save_path_root + "/" + str(i) +".png"
        cv2.imwrite(savepath,c)

#parsing input arguments
parser = argparse.ArgumentParser(description='Face Detection from Video')
parser.add_argument('gpu_or_cpu', metavar='gpu_or_cpu',
                    help='Device to use for inference, for gpu enter gpu else enter cpu')
parser.add_argument('video_location', metavar='video_location',
                    help='Location of video')
parser.add_argument('save_location', metavar='save_location',
                    help='Location where to save files')
args = parser.parse_args()

#declaring models
mtcnn = MTCNN(image_size=112, margin=0, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
if args.gpu_or_cpu == "gpu":
    resnet = resnet.to("cuda:0")

#reads video file
video_file = cv2.VideoCapture(args.video_location)

#stores video frames as list of images
video_frames = []
while(video_file.isOpened()):
    ret, frame = video_file.read()
    if frame is None:
        break
    video_frames.append(frame)

#makes directory to save images
os.makedirs(args.save_location)

#getting faces from input video stream
final_faces_list = []
for frame in video_frames:
    images = face_extractor_from_frame(frame)
    refined_images = []
    if images is None:
        continue
    else:
        refined_images.append(images)
    for image in images:
        if image is None:
            continue
        else:
            final_faces_list.append(image)

#prints number of faces detected in complete video
print("Number of faces detected: ",len(final_faces_list))

#getting feature vectors from detected faces
#default batch size is set to 8 change the batch size below
batch_size = 8
face_features = np.ndarray((len(final_faces_list),512), dtype=np.float32)
if args.gpu_or_cpu == "gpu":
    for start_index in tqdm(range(0, len(final_faces_list),batch_size)):
        end_index = min(start_index+batch_size, len(final_faces_list))
        imgs = final_faces_list[start_index:end_index]
        imgs = torch.stack((imgs))
        imgs = imgs.to("cuda:0")
        features = resnet(imgs)
        face_features[start_index:end_index] = features.detach().cpu().numpy()
else:
    for start_index in tqdm(range(0, len(final_faces_list),batch_size)):
        end_index = min(start_index+batch_size, len(final_faces_list))
        imgs = final_faces_list[start_index:end_index]
        imgs = torch.stack((imgs))
        features = resnet(imgs)
        face_features[start_index:end_index] = features.detach().numpy()

min_sam = int(len(final_faces_list)*0.1)
clt = DBSCAN(min_samples=min_sam, metric="euclidean", n_jobs=4)
clt.fit(face_features)

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("Number of Unique faces detected: ",numUniqueFaces)

for labelID in labelIDs:
    if labelID==-1:
        continue
    else:
        savepath = args.save_location + "/" + str(labelID)
        os.makedirs(savepath)
        idxs = np.where(clt.labels_ == labelID)[0]
        for i in idxs:
            temp_img = final_faces_list[i]
            saveimgpath = args.save_location + "/" + str(labelID) + "/" + str(i) + ".png"
            temp_img = temp_img.numpy().transpose(1,2,0)
            temp_img = (temp_img+1)/2
            temp_img = temp_img*255.0
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(saveimgpath,temp_img)