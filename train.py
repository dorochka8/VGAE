import torch
import torch.nn.functional as F
import torch_geometric

from sklearn.metrics import roc_auc_score, average_precision_score


def train(model, x, edge_index, optimizer, epochs, device):
  losses, roc_aucs = [], []
  adj  = create_adj(x, edge_index)
  norm = create_norm(adj)
  
  x = x.to(device)
  edge_index = edge_index.to(device)

  for epoch in range(epochs):
      model.train()
      optimizer.zero_grad()

      z, mu, log_std = model(x, edge_index)
      neg_edges = torch_geometric.utils.negative_sampling(edge_index=edge_index, 
                                                        num_nodes=x.shape[0], 
                                                        num_neg_samples=edge_index.size(1),
                                                        )
      neg_pos_edge_index = torch.cat([edge_index, neg_edges], dim=-1)

      edge_logits = model.decode(z, neg_pos_edge_index)
      edge_labels = torch.cat([torch.ones(edge_index.size(1)), torch.zeros(neg_edges.size(1))], dim=0).to(device)

      loss = loss_fn(edge_logits, edge_labels, mu, log_std, norm)
      losses.append(loss.item())

      roc_auc = roc_auc_score(edge_labels.cpu().detach().numpy(), edge_logits.cpu().detach().numpy())
      roc_aucs.append(roc_auc)

      ap_score = average_precision_score(edge_labels.cpu().detach().numpy(), edge_logits.cpu().detach().numpy())

      if (epoch + 1) % 50  == 0 or epoch == 0:
          print(f'Epoch: {epoch+1}, \tloss: {loss.item()}, \troc_auc: {roc_auc} \tap: {ap_score}')
      
      loss.backward()
      optimizer.step()

  return losses, roc_aucs


def loss_fn(y_pred, y_true, mu, log_std, norm):
    KLD = torch.mean(0.5 * torch.sum(1 + 2*log_std - mu**2 - (log_std**2).exp(), dim=1)) / y_pred.size(0)
    BCE = norm * F.binary_cross_entropy_with_logits(y_pred.view(-1), y_true.view(-1))
    return BCE -  KLD


def create_adj(x, edge_index):
    adj = torch.eye(x.size()[0], x.size()[0])
    for i, j in zip(edge_index[0], edge_index[1]):
        adj[i.item(), j.item()] += 1 

    return adj


def create_norm(adj):
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return norm