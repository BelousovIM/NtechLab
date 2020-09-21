import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

from PIL import Image
import os, sys
from tqdm import tqdm

import shutil

import json


channels = 3
class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(channels, 64, 2, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3),
                                        nn.Conv2d(64, 32, 2, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3),)
        self.linear_layers = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,2), nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def resize(path):
    dirs = os.listdir(path)
    for item in tqdm(dirs):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((32,32), Image.ANTIALIAS)
            imResize.save('new_images_wb/' + item, 'JPEG', quality=90)
        clear_output(True)
        
def process(path):
    A = {0:'male', 1:'female'}
    # Приводим к одному размеру
    if 'new_images_wb' in os.listdir():
        shutil.rmtree('new_images_wb')
    os.mkdir('new_images_wb')
    resize(path)
    #Качаем модель
    PATH = 'entire_model.pt'
    model = torch.load(PATH, map_location={'cuda:0': 'cpu'})
    model.eval()
    #Создаем X
    dirs = ['new_images_wb/' + i for i in os.listdir('new_images_wb')]
    #extracting images
    X = []
    for d in dirs:
        image = plt.imread(d)
        X.append(image.reshape((3,32,32)))
    X = np.array(X).astype('float32')/255
    X = torch.from_numpy(X)
    y_predict = np.argmax(model(X).cpu().data.numpy(), axis=1)
    y_predict = [A[i] for i in y_predict]
    with open('drive/My Drive/process_results.json', "w", encoding="utf-8") as file:
        json.dump(dict(zip(os.listdir('new_images_wb'), y_predict)), file)
        
if __name__ == '__main__':
    process(sys.argv[1])
