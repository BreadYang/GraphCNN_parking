import torch
import torch.nn as nn

import numpy as np
import networkx as nx
import pickle

from collections import OrderedDict
from callbacks import EarlyStopping

#
class one_dateset():
  def __init__(self, x, name, typ, config):
    self.x = x
    self.name = name
    self.typ = typ
    self.config = config
    self.config['type'] = typ
  
  def __len__(self):
    return self.x.size(0)
    
  def __getitem__(self, index):
    return self.x[index]

#
class traffic_datesets():
  def __init__(self):
    self.dataset_dict = OrderedDict()

  def register_dateset(self, x, name, typ, config):
    self.dataset_dict[name] = one_dateset(x, name, typ, config)

  def register_dateset(self, dataset):
    self.dataset_dict[dataset.name] = dataset

  def register_target(self, y):
    self.y = y

  def generate_config_list(self):
    config_list = list()
    for dataset in self.dataset_dict.itervalues():
      config_list.append(dataset.config)
    return config_list

  def get_data_list(self, indices = None):
    x_list = list()
    if indices is None:
      for dataset in self.dataset_dict.itervalues():
        x_list.append(dataset.x)
      return x_list
    for dataset in self.dataset_dict.itervalues():
      part_x = dataset.x[indices]
      x_list.append(part_x)
    return x_list

  def get_y(self, indices = None):
    if indices is None:
      return self.y
    return self.y[indices]

  def get_num_data(self):
    return len(self.dataset_dict[self.dataset_dict.keys()[0]])

  def to_cuda(self):
    for name, dataset in self.dataset_dict.iteritems():
      dataset.x = dataset.x.cuda()
      if 'L' in dataset.config.keys():
        self.dataset_dict[name].config['L'] = self.dataset_dict[name].config['L'].cuda()
    self.y = self.y.cuda()

  def is_ok(self):
    pass

#
class D_embedding(nn.Module):
  def __init__(self, D, num_hidden, num_output):
    super(D_embedding, self).__init__()
    self.D = D
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.model = nn.Sequential(
                  nn.Linear(D, self.num_hidden),
                  nn.ReLU(),
                  nn.Linear(self.num_hidden, self.num_output)
                  )
  def forward(self, x):
    return self.model(x)

#3,
class TD_embedding(nn.Module):
  def __init__(self, T, D, num_layer, num_output, dropoutLSTM):
    super(TD_embedding, self).__init__()
    self.D = D
    self.T = T
    self.num_layer = num_layer
    self.num_output = num_output
    self.model = nn.LSTM(D, self.num_output, self.num_layer)

  def forward(self, x):
    x = x.permute(1, 0, 2)
    output_seq, _ = self.model(x)
    last_output = output_seq[-1]
    return last_output

#
class VD_embedding(nn.Module):
  def __init__(self, V, D, L, F, C, O, num_output, K = 2):
    super(VD_embedding, self).__init__()
    self.V = V # number of node
    self.D = D # number of features
    self.L = L # L matrix
    self.K = K # number of order for cheb approx
    self.F = F # number of output filter
    # self.C = C # number of Cheb
    self.O = O # num of Cov process
    for i in range(O):
      self.L = torch.matmul(self.L, L)
    self.num_output = num_output
    self.W = torch.nn.Parameter(torch.FloatTensor(self.K, self.F))
    self.b = torch.nn.Parameter(torch.FloatTensor(1, 1, self.D, self.F))
    self.after_model = nn.Sequential(
                        nn.Linear(self.D * self.F, self.num_output)
                        )
  def forward(self, x):
    filtered_x = _chebyshev(x, self.K, self.L)
    tmp = torch.matmul(filtered_x.view(-1, self.K), self.W)
    tmp2 = tmp.view(-1, self.V, self.D, self.F) + self.b
    return self.after_model(tmp2.view(-1, self.V, self.D * self.F))

#
def _chebyshev(x, K, L):
  orig_size = x.size()
  n = orig_size[0]
  filtered_x = torch.autograd.Variable(torch.FloatTensor(n, orig_size[1], orig_size[2], K))
  if x.is_cuda:
    filtered_x = filtered_x.cuda()
  # print x.is_cuda, L.is_cuda, filtered_x.is_cuda
  for i in range(n):
    for j in range(K):
      if j == 0:
        filtered_x[i, :, :, j] = x[i, :, :]
      elif j == 1:
        filtered_x[i, :, :, j] = torch.matmul(L, x[i, :, :])
      else:
        filtered_x[i, :, :, j] = 2 * torch.matmul(L, filtered_x[i, :, :, j-1]) - filtered_x[i, :, :, j-2]
  return filtered_x

#
class VTD_embedding(nn.Module):
  def __init__(self, T, num_layer, V, D, L, F, C, O, num_output, num_hidden, K = 2, dropoutLSTM = 0.0):
    super(VTD_embedding, self).__init__()
    self.V_model = VD_embedding(V, D, L, F, C, O, num_hidden)
    self.T_model = TD_embedding(T, num_hidden, num_layer, num_output, dropoutLSTM)
    self.V = V
    self.D = D
    self.L = L
    self.k = K
    self.T = T
    self.O = O
    self.num_hidden = num_hidden
    self.num_output = num_output

  def forward(self, x):
    x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.V, self.D)
    filtered_x = self.V_model(x).view(-1, self.T, self.V, self.num_hidden)
    # print filtered_x.size()
    v_input = filtered_x.permute(0, 2, 1, 3).contiguous().view(-1, self.T, self.num_hidden)
    # print v_input.size()
    return self.T_model(v_input).view(-1, self.V, self.num_output)


#
class parking_prediction(nn.Module):
  def __init__(self, V, data_config, after_config, dropout_rate):
    super(parking_prediction, self).__init__()
    self.data_config = data_config
    self.V = V
    self.dropout = dropout_rate
    self.embedding_dict = OrderedDict()
    self.init_embedding()
    self.build_aftermodel(after_config, dropout_rate)
    self.init_parameters()

  def init_embedding(self):
    self.total_output = 0
    for config in self.data_config:
      if config['type'] == 'D':
        self.embedding_dict[config['name']] = D_embedding(config['D'], config['num_hidden'], config['num_output'])
      if config['type'] == 'TD':
        self.embedding_dict[config['name']] = TD_embedding(config['T'], config['D'], config['num_layer'],  config['num_output'])
      if config['type'] == 'VD':
        self.embedding_dict[config['name']] = VD_embedding(config['V'], config['D'], config['L'], config['F'], config['C'], config['O'], config['num_output'])
      if config['type'] == 'VTD':
        self.embedding_dict[config['name']] = VTD_embedding(config['T'], config['num_layer'], config['V'], 
                                                                config['D'], config['L'], config['F'], config['C'], 
                                                                config['O'],config['num_output'], config['num_hidden'], config['dropoutLSTM'])
      if config['type'] == 'VTD':
        self.total_output += config['num_output']*config['V']
      else:
        self.total_output += config['num_output']

  def build_aftermodel(self, after_config, dropout_rate):
    self.after_model = nn.Sequential(
                        nn.Linear(self.total_output, after_config['num_hidden']),
                        nn.Dropout2d(p=self.dropout, inplace=False),
                        nn.ReLU(),
                        nn.Linear(after_config['num_hidden'], self.V)
                        )

  def init_parameters(self):
    for name, embedding in self.embedding_dict.iteritems():
      for p in embedding.parameters():
        if p.ndimension()  < 2:
          torch.nn.init.constant(p, 0)
        else:
          torch.nn.init.xavier_uniform(p)

    for p in self.after_model.parameters():
      if p.ndimension()  < 2:
        torch.nn.init.constant(p, 0)
      else:
        torch.nn.init.xavier_uniform(p)

  def get_all_parameters(self):
    p_list = list()
    for p in self.after_model.parameters():
      p_list.append(p)
    for name, embedding in self.embedding_dict.iteritems():
      for p in embedding.parameters():
        p_list.append(p)
    return p_list

  def set_callbacks(self, callback):
    super(nn.Module, self).set_callbacks(callback)


  def forward(self, x_list):
    num_data = len(x_list)
    embedding_name_list = list(self.embedding_dict.keys())
    embedded_list = list()
    for i in range(num_data):
      x = x_list[i]
      embedding_name = embedding_name_list[i]
      embedding = self.embedding_dict[embedding_name]
      embeded_x = embedding(x)
      if len(embeded_x.size()) == 3:
        orig_size = embeded_x.size()
        embeded_x = embeded_x.unsqueeze(1).view(-1, orig_size[1]*orig_size[2])
      embedded_list.append(embeded_x)
    embedded_all = torch.cat(embedded_list, dim=1).view(-1, self.total_output)
    return self.after_model(embedded_all).view(-1, self.V)


#
def train(train_set, test_set, V, after_config, nl, batch_size = 32, num_epoch = 10, dropout_rate = 0.3,
            learning_rate = 0.1, verbose = True, save = None, use_GPU = True):
  if use_GPU:
    train_set.to_cuda()
    test_set.to_cuda()

  config_list = train_set.generate_config_list()
  model = parking_prediction(V, config_list, after_config, dropout_rate)

  if use_GPU:
    model = model.cuda()
    for name, embedding in model.embedding_dict.iteritems():
      model.embedding_dict[name] = embedding.cuda()

  optimizer = torch.optim.Adam(model.get_all_parameters(), lr = learning_rate, weight_decay = 1e-4)
  loss_fn = torch.nn.MSELoss()

  if verbose: 
    print "Start Training..."
  total_train = train_set.get_num_data()
  record = [list(), list()]
  record[0].append(loss_fn(model(train_set.get_data_list()), train_set.get_y()).cpu().data.numpy()[0])
  record[1].append(loss_fn(model(test_set.get_data_list()), test_set.get_y()).cpu().data.numpy()[0])
  for i in range(num_epoch):
    total_train_loss = 0.0
    seq = np.random.permutation(total_train)
    train_sample_list = np.array_split(seq, len(seq) / batch_size)
    for sample_ind in train_sample_list:
      t_sample_ind = torch.LongTensor(sample_ind)
      if use_GPU:
        t_sample_ind = t_sample_ind.cuda()
      sample_x = train_set.get_data_list(t_sample_ind)
      sample_y = train_set.get_y(t_sample_ind)
      predicted_y = model.forward(sample_x)
      loss = loss_fn(predicted_y, sample_y)
      total_train_loss += loss.data.cpu().numpy()[0] * len(sample_ind)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss = total_train_loss / total_train
    test_loss = loss_fn(model(test_set.get_data_list()), test_set.get_y()).cpu().data.numpy()[0]
    if verbose:
      print "Epoch ", i, "Training loss: ", train_loss, "Testing loss: ", test_loss
    if verbose:
      print "Epoch ", i, "Testing loss: ", test_loss
    record[0].append(train_loss)
    record[1].append(test_loss)

  if save is not None:
    assert(type(save) == str)
    pickle.dump((record, n), open(save, 'wb'))
  return (model, record)


class winsorize_normalizer():
  def __init__(self, lower = 5, upper = 95):
    self.mins = list()
    self.maxs = list()
    self.lower = lower
    self.upper = upper

  def fit(self, X):
    X_numpy = X.data.cpu().numpy()
    D = X_numpy.shape[-1]
    for d in range(D):
      tmp_X = X_numpy[..., d]
      self.mins.append(np.percentile(tmp_X, self.lower))
      self.maxs.append(np.percentile(tmp_X, self.upper))

  def transform(self, X):
    D = X.size()[-1]
    assert(D == len(self.mins))
    new_X = X.clone()
    for d in range(D):
      new_X[..., d] = torch.clamp(X[..., d], float(self.mins[d]), float(self.maxs[d]))
    return new_X

  def reverse(self, X):
    return X

class std_normalizer():
  def __init__(self, eps = 0.1):
    self.means = list()
    self.stds = list()
    self.eps = eps

  def fit(self, X):
    X_numpy = X.data.cpu().numpy()
    D = X_numpy.shape[-1]
    for d in range(D):
      tmp_X = X_numpy[..., d]
      self.means.append(np.mean(tmp_X))
      self.stds.append(np.std(tmp_X))

  def transform(self, X):
    D = X.size()[-1]
    assert(D == len(self.means))
    new_X = X.clone()
    for d in range(D):
      new_X[..., d] = (X[..., d] - float(self.means[d])) / (float(self.stds[d]) + self.eps)
    return new_X

  def reverse(self, X):
    D = X.size()[-1]
    assert(D == len(self.means))
    new_X = X.clone()
    for d in range(D):
      new_X[..., d] = X[..., d] * (float(self.stds[d]) +  self.eps) +  float(self.means[d])
    return new_X

class minmax_normalizer():
  def __init__(self, eps = 0.01):
    self.mins = list()
    self.maxs = list()
    self.eps = eps

  def fit(self, X):
    X_numpy = X.data.cpu().numpy()
    D = X_numpy.shape[-1]
    for d in range(D):
      tmp_X = X_numpy[..., d]
      self.mins.append(np.min(tmp_X))
      self.maxs.append(np.max(tmp_X))

  def transform(self, X):
    D = X.size()[-1]
    assert(D == len(self.mins))
    new_X = X.clone()
    for d in range(D):
      new_X[..., d] = (((X[..., d] - float(self.mins[d])) / (float(self.maxs[d] - self.mins[d]) + self.eps))) * 2 - 1
    return new_X    

  def reverse(self, X):
    D = X.size()[-1]
    assert(D == len(self.mins))
    new_X = X.clone()
    for d in range(D):
      new_X[..., d] = ((X[..., d] + 1) / 2)  * (float(self.maxs[d] - self.mins[d]) + self.eps) +  float(self.mins[d])
    return new_X

class normalizers():
  def __init__(self):
    self.normalizer_dict = OrderedDict()

  def build_normalizer(self, name, norm_list, X):
    tmp_X = X.clone()
    normer_list = list()
    for norm in norm_list:
      if norm == 'wins':
        n = winsorize_normalizer()
        n.fit(tmp_X)
        normer_list.append(n)
      if norm == 'std':
        n = std_normalizer()
        n.fit(tmp_X)
        normer_list.append(n)  
      if norm == 'minmax':
        n = minmax_normalizer()
        n.fit(tmp_X)
        normer_list.append(n)
      tmp_X = n.transform(tmp_X)
    self.normalizer_dict[name] = normer_list

  def transform(self, name, X):
    normer_list = self.normalizer_dict[name]
    tmp_X = X
    for normer in normer_list:
      tmp_X = normer.transform(tmp_X)
    return tmp_X

  def reverse(self, name, X):
    normer_list = self.normalizer_dict[name]
    tmp_X = X
    for normer in reversed(normer_list):
      tmp_X = normer.reverse(tmp_X)
    return tmp_X





# #
# class parking_prediction(nn.Module):
#   def __init__(self, V, data_config, after_config):
#     super(parking_prediction, self).__init__()
#     self.data_config = data_config
#     self.V = V
#     self.embedding_dict = OrderedDict()
#     self.init_embedding()
#     self.build_aftermodel(after_config)
#     self.init_parameters()

#   def init_embedding(self):
#     self.total_output = 0
#     for config in self.data_config:
#       if config['type'] == 'D':
#         self.embedding_dict[config['name']] = D_embedding(config['D'], config['num_hidden'], config['num_output'])
#       if config['type'] == 'TD':
#         self.embedding_dict[config['name']] = TD_embedding(config['T'], config['D'], config['num_layer'],  config['num_output'])
#       if config['type'] == 'VD':
#         self.embedding_dict[config['name']] = VD_embedding(config['V'], config['D'], config['L'], config['F'], config['num_output'])
#       if config['type'] == 'VTD':
#         self.embedding_dict[config['name']] = VTD_embedding(config['T'], config['num_layer'], config['V'], 
#                                                                 config['D'], config['L'], config['F'], 
#                                                                 config['num_output'], config['num_hidden'])
#       self.total_output += config['num_output']

#   def build_aftermodel(self, after_config):
#     self.after_model = nn.Sequential(
#                         nn.Linear(self.total_output, after_config['num_hidden']),
#                         nn.ReLU(),
#                         nn.Linear(after_config['num_hidden'], 1)
#                         )

#   def init_parameters(self):
#     for name, embedding in self.embedding_dict.iteritems():
#       for p in embedding.parameters():
#         if p.ndimension()  < 2:
#           torch.nn.init.constant(p, 0)
#         else:
#           torch.nn.init.xavier_uniform(p)

#     for p in self.after_model.parameters():
#       if p.ndimension()  < 2:
#         torch.nn.init.constant(p, 0)
#       else:
#         torch.nn.init.xavier_uniform(p)

#   def get_all_parameters(self):
#     p_list = list()
#     for p in self.after_model.parameters():
#       p_list.append(p)
#     for name, embedding in self.embedding_dict.iteritems():
#       for p in embedding.parameters():
#         p_list.append(p)
#     return p_list

#   def forward(self, x_list):
#     num_data = len(x_list)
#     embedding_name_list = list(self.embedding_dict.keys())
#     embedded_list = list()
#     for i in range(num_data):
#       x = x_list[i]
#       embedding_name = embedding_name_list[i]
#       embedding = self.embedding_dict[embedding_name]
#       embeded_x = embedding(x)
#       if len(embeded_x.size()) < 3:
#         orig_size = embeded_x.size()
#         embeded_x = embeded_x.unsqueeze(1).expand(orig_size[0], self.V, orig_size[1])
#       embedded_list.append(embeded_x)
#     embedded_all = torch.cat(embedded_list, dim=2).view(-1, self.total_output)
#     return self.after_model(embedded_all).view(-1, self.V)
#     