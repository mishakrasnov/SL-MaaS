from typing import List
from fastapi import FastAPI
import torch
import torch.nn.functional as F
import numpy as np
import gdown

# PyTorch Geometric imports

import torchvision.models as models



model = models.resnet18(num_classes=24)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

gdown.download('https://drive.google.com/uc?id=1cBPywP7OgrXFvHuEpEG-DPM9rp8wXM8n')
checkpoint = torch.load('resnet_weights.pt', map_location=torch.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def adjToEdgidx(adj_mat):
    edge_index = torch.from_numpy(adj_mat).nonzero().t().contiguous()
    row, col = edge_index
    edge_weight = adj_mat[row, col]  # adj_mat[row, col]
    return edge_index, edge_weight


app = FastAPI()


@app.post("/NVG")
async def echo(data: List[List[float]]):
    # print(data[:5])
    data = torch.tensor(data)
    out = model(data)
    pred = out.argmax(dim=1)
    # print(pred)

    return {"pred": str(pred)}