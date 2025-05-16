from typing import List
from fastapi import FastAPI
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, CrossEntropyLoss
import numpy as np
from ts2vg import NaturalVG
import gdown

# PyTorch Geometric imports
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, ChebConv, global_sort_pool

import torchvision.models as models

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    pin_memory = True
else:
    device = torch.device('cpu')
    pin_memory = False

print("Current device: ", torch.cuda.current_device())  
print("Current device: ", torch.cuda.is_available())


model = models.resnet18()

gdown.download('https://drive.google.com/uc?id=1cBPywP7OgrXFvHuEpEG-DPM9rp8wXM8n')
checkpoint = torch.load('resnet_weights.pt')

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)  

def adjToEdgidx(adj_mat):
    edge_index = torch.from_numpy(adj_mat).nonzero().t().contiguous()
    row, col = edge_index
    edge_weight = adj_mat[row, col]  # adj_mat[row, col]
    return edge_index, edge_weight


app = FastAPI()


@app.post("/NVG")
async def echo(data: List[List[float]]):
    # print(data[:5])
    data = np.array(data).reshape(300)
    g = NaturalVG(weighted="distance")
    g.build(data)
    adj_mat = g.adjacency_matrix(use_weights=True, no_weight_value=0)
    edge_index, edge_weight = adjToEdgidx(adj_mat)
    inp = Data(x=torch.unsqueeze(torch.tensor(data, dtype=torch.double), 1), edge_index=edge_index,
               edge_attr=torch.unsqueeze(torch.tensor(edge_weight, dtype=torch.double), 1))
    # print(inp)
    out = model(inp)
    pred = out.argmax(dim=1)
    # print(pred)

    return {"pred": str(pred)}