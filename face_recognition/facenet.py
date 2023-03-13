import cv2
import time
import sys
sys.path.append(".")
from .models.mtcnn import MTCNN
from .models.inception_resnet_v1 import InceptionResnetV1
import torch
import math

from PIL import Image

import face_recognition.save_embedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

load_data = torch.load('deployment/assets/embedding.pt')
embedding_list = load_data[0]
name_list = load_data[1]
print("[INFO] Loaded Embedding...!")

def recognize(img, get_axes=False):
    print("[INFO] Checking image...")
    name = "Unknown"
    boxes = None
    out_dist = []

    if img is not None:
        t0 = time.time()
        face_cropped_list, prob_list = mtcnn(img, return_prob=True)
        t1 = time.time()
        print("[INFO] MTCNN time: {}s".format(round(t1 - t0, 2)))

        if face_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)

            for i, prob in enumerate([prob_list]):
                if prob > 0.90:
                    emb = resnet(face_cropped_list.to(device).unsqueeze(0)).detach()

                    dist_list = []  # list of distance from database

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)
                        out_dist.append(dist)

                    # print(dist_list)
                    min_dist = min(dist_list)   # minimum distant value
                    min_dist_idx = dist_list.index(min_dist)
                    name = name_list[min_dist_idx]
            t2 = time.time()
            
            print("[INFO] Recognition time: {}s".format(round(t2 - t1, 2)))

        if get_axes:
            return name, boxes
        return name   
    return None, None

def test():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print('[ERROR] Fail to grab frame, please try again!')
            break

        img = Image.fromarray(frame)
        face_cropped_list, prob_list = mtcnn(img, return_prob=True)
        #print(face_cropped_list, prob_list)

        if face_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            for i, prob in enumerate([prob_list]):
                if prob > 0.9:
                    t0 = time.time()
                    emb = resnet(face_cropped_list.to(device).unsqueeze(0)).detach()
                    dist_list = []  # list of distance from database

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)   # minimum distant value
                    min_dist_idx = dist_list.index(min_dist)
                    t1 = time.time()
                    name = name_list[min_dist_idx]
                    box = boxes[i]

                    if min_dist < 0.9:
                        frame = cv2.putText(frame, name + " " + str(round(min_dist, 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)

                    frame = cv2.putText(frame, "FPS: " + str(math.ceil(1 / round(t1 - t0, 2))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)


        cv2.imshow("IMG", frame)
        k = cv2.waitKey(1)
        if k%256==27: # ESC
            print('[INFO] Esc pressed, closing...')
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()