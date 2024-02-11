import torch
import torch_geometric
import matplotlib.pyplot as plt

from config import config
from model import GVAE
from train import train

epochs = config['epochs']
device =  'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = './'
dataset = torch_geometric.datasets.Planetoid(data_dir, name='Cora')
data = dataset[0]

x = data.x
edge_index = data.edge_index
in_channels  = x.shape[1]
out_channels = 2

model = GVAE(in_channels, out_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

losses, roc_aucs = train(model, x, edge_index, optimizer, epochs, device)

plt.plot(losses)
plt.show()

plt.plot(roc_aucs)
plt.show()
