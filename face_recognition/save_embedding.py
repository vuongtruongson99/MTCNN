from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
from tqdm import tqdm
sys.path.append(".")

worker = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print('[INFO] Running on device: {}'.format(device))

# Load face detector
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Load face recognizer
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder("../deployment/assets/database")
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=worker)

idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

name_list = []          # list of name corresponding to cropped photos
embedding_list = []     # embeding matrix (vector 512)

#print(loader)
for img, idx in tqdm(loader):
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob >= 0.9:
        emb = resnet(face.to(device).unsqueeze(0))
        embedding_list.append(emb)
        name_list.append(idx_to_class[idx])

# Save data
data = [embedding_list, name_list]
torch.save(data, '../deployment/assets/embedding.pt')
print("[INFO] SAVE SUCCESSFUL...!")