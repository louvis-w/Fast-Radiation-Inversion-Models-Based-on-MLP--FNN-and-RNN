import torch
from torch import nn
from torch.utils import data
from data_loader import get_some_data, std_mean_tb_dict
from util import try_gpu
import time
from matplotlib import pyplot as plt
import numpy as np

dropout1 = 0.1
dropout2 = 0.1
dropout3 = 0.1

device='cuda:0'

var_list = ['wc_rain', 'wc_snow', 'wc_ice', 'wc_cloud', 'temperature', 'level_pressure','q', 
            'u10', 'v10', 'skt', 'zenith_angle', 'azimuth_angle']

sensor_list = ['mwhs_fy3a', 'mwts_fy3a', 'mwri_fy3a']
#sensor_list = ['mwri_fy3a']

nch_sensor = {'mwhs_fy3a':5,
             'mwts_fy3a':4, 
             'mwri_fy3a':10}

class MLP(nn.Module):
    def __init__(self, in_features, out_features, n_hidden):
        super().__init__()
        self.hidden1 = nn.Linear()


def get_net(in_features, out_features, n_hidden, device):
    net = nn.Sequential(nn.Linear(in_features, n_hidden),
                        nn.ReLU(),
                        #nn.Dropout(dropout1),
#                        nn.Linear(n_hidden, n_hidden),
#                        nn.ReLU(),
                        #nn.Dropout(dropout2),
                        nn.Linear(n_hidden, n_hidden),
                        nn.ReLU(),
                        #nn.Dropout(dropout3),
                        nn.Linear(n_hidden, out_features),
                        nn.Tanh())
    net.to(device=device)
    return net

def save_net(net, filename):
    torch.save(net.state_dict(), filename)

def load_net(filename, in_features, out_features, n_hidden, device):
    net = nn.Sequential(nn.Linear(in_features, n_hidden),
                        nn.ReLU(),
                        #nn.Dropout(dropout1),
#                        nn.Linear(n_hidden, n_hidden),
#                        nn.ReLU(),
                        #nn.Dropout(dropout2),
                        nn.Linear(n_hidden, n_hidden),
                        nn.ReLU(),
                        #nn.Dropout(dropout3),
                        nn.Linear(n_hidden, out_features),
                        nn.Tanh())
    net.load_state_dict(torch.load(filename))
    net.to(device=device)
    return net

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, train_features, train_labels, test_features, test_labels, 
            num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    
    train_features_device = train_features.to(device)
    train_labels_device = train_labels.to(device)
    test_features_device = test_features.to(device)
    test_labels_device = test_labels.to(device)
    
    train_iter = load_array( (train_features_device, train_labels_device), batch_size)
    optimizer = torch.optim.Adam( net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss = nn.MSELoss()
    t0 = time.time()


    for epoch in range(num_epochs):
        vloss = 0.
        nloss = 0
        for X, y in train_iter:
            #X = X.to(device)
            #y = y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            vloss += l.item()
            nloss += 1
        with torch.no_grad():
#            print('-----------------------------------')
#            print(net(train_features[0,:]))
#            print(train_labels[0,:])

            train_ls.append( torch.sqrt( loss( net(train_features_device), train_labels_device)) )
            if test_labels is not None:
                test_ls.append( torch.sqrt( loss( net(test_features_device), test_labels_device)) )
         
        t1 = time.time()
        print('epoch %4d, time=%8f, train_loss=%f, test_loss=%f'% (epoch, t1-t0, train_ls[-1], test_ls[-1]) )
#        print('epoch %4d, time=%8f, training_loss=%8f'% (epoch, t1-t0, vloss/nloss) )
        t0 = t1
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx= slice(j*fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx,:], y[idx,:]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epocs, learning_rate, weight_decay, batch_size, device):
    train_l_sum, valid_l_sum = 0, 0

    in_features = X_train.shape[1]
    out_features = y_train.shape[1]
    n_hidden = 64
    for i in range(1, k):
        data = get_k_fold_data(k, i, X_train, y_train)
        #print(data)
        
        net = get_net(in_features, out_features, n_hidden, device=device)
        train_ls, valid_ls = train( net, *data, num_epocs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print(  f'fold{i + 1}, training rmse={float(train_ls[-1]):f}, '
                f'validating rmse={float(valid_ls[-1]):f}')
    return train_l_sum/k, valid_l_sum/k

def run_training(n_hidden):
    k = 5
    num_epochs = 100
    lr = 0.00003
    weight_decay = 0.00003
    batch_size = 128
    device='cuda:0'
    
    #sensor_list = ['mwri_fy3a']
    idx_ch = slice(9,10)
    filename1 = '/disk1/yxl232/data/nn_rt/training/ERA5-extracted-0.25-20100201-0000.nc'
    train_features1, train_labels1 = get_some_data(filename1, var_list, sensor_list)
    train_features = train_features1
    #train_labels = train_labels1[:,idx_ch]
    train_labels = train_labels1

    filename2 = '/disk1/yxl232/data/nn_rt/training/ERA5-extracted-0.25-20110201-0000.nc'
    test_features1, test_labels1 = get_some_data(filename2, var_list, sensor_list)
    test_features = test_features1
    #test_labels = test_labels1[:,idx_ch]
    test_labels = test_labels1
    #print(train_features)
    #print(train_labels)
    #train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, device=device)

    in_features = train_features.shape[1]
    out_features = train_labels.shape[1]
    print(in_features)
    print(out_features)

    net = get_net(in_features, out_features, n_hidden, device=device)
    train_ls, valid_ls = train( net, train_features, train_labels, test_features, test_labels, 
                                num_epochs, lr, weight_decay, batch_size)
    
#    print(  f'{k}-fold: main training rmse: {float(train_l):f}, '
#            f'main validating rmse: {float(valid_l):f}')

    filename = 'mlp.params-300epoch'
    save_net(net, filename)
    net2 = load_net(filename, in_features, out_features, n_hidden, 'cpu')

    #X = torch.tensor(test_features[0:1,:])
    X = test_features[0:1,:]
    #X.to(device)
    net.to('cpu')

    print( net(X))
    print( net2(X))


def run_diff(n_hidden):
    

#    filename1 = '/disk1/yxl232/data/nn_rt/training/ERA5-extracted-0.25-20100201-0000.nc'
#    train_features1, train_labels1 = get_some_data(filename1, var_list, sensor_list)
#    train_features = train_features1
#    train_labels = train_labels1

    filename2 = '/disk1/yxl232/data/nn_rt/training/ERA5-extracted-0.25-20110201-0000.nc'
    test_features1, test_labels1 = get_some_data(filename2, var_list, sensor_list)
    test_features = test_features1
    test_labels = test_labels1

    in_features = test_features.shape[1]
    out_features = test_labels.shape[1]

    filename = 'mlp.params-300epoch'
    net2 = load_net(filename, in_features, out_features, n_hidden, 'cpu')

    y = net2(test_features)
    diff = (y - test_labels).detach().numpy()
    mean_scaled = diff.mean(axis=0)
    std_scaled =  diff.std(axis=0)
    
    return mean_scaled, std_scaled
    

def run_ploting(mean=None, std=None, mean_scaled=None, std_scaled=None):
    if mean is None:
        mean = mean_scaled * 320
        std = std_scaled * 320
    nch = mean.shape[0]

    print(mean.tolist())
    print(std.tolist())

    idx = 0
    for sensor in sensor_list:
        for ich in range( nch_sensor[sensor] ):
            mean_ch, std_ch = std_mean_tb_dict['tb_'+sensor][ich]
            mean[idx] /= mean_ch
            std[idx] /= mean_ch
            idx += 1

    print(mean.tolist())
    print(std.tolist())

def save_later():

    ''''
    hidden_layers=3, n_hidden = 256, epochs = 30, 
    [-0.31063375 -0.40106994 -0.08925882 -0.18728676 -0.13048707 -0.11070438
      0.01839026 -0.12724367 -0.2188927   0.59721553  0.50699115  0.6228513
      0.6781035   0.34172332  0.39596105 -0.02548948 -0.01739007 -0.37849382
    -0.23192374]
    [ 2.5800517 2.6471517 3.8537385 2.679639  2.338077  1.6313026 1.6013427
      1.6661539 1.4531167 1.5404711 1.463189  1.4433408 1.3524753 1.5258384
      1.4924412 1.71774   1.7444295 1.9689105 2.1301563]
    
    hidden_layers=3, n_hidden = 128, epochs = 30, 
    [ 0.03225781 -0.00729531  0.26655358  0.12977493  0.0223471  -0.17568323
      0.04008387 -0.00691236 -0.02264911  0.22707774 -0.00125483  0.12455943
     -0.06223147  0.1538853   0.00335282 -0.10701308 -0.2714685   0.01968138
      0.08240204]
    [ 2.6380112 2.7210984 3.8812459 2.7323117 2.4454656 1.6579833 1.627631
      1.6836214 1.4910973 1.6219058 1.4771492 1.415556  1.3586705 1.5154405
      1.500454  1.7124474 1.7169852 2.0379086 2.1891866]
    
    hidden_layers=4, n_hidden = 128, epochs = 30, 
    [ 6.1955709e-02  3.8903665e-02  3.3686243e-02  2.3849592e-02
     -4.2724483e-02 -3.6229346e-02  6.0856074e-02  1.6359264e-01
      1.7840119e-01  2.3699626e-01  5.5216676e-01  6.2332507e-02
      2.5788876e-01  4.5998083e-04  1.7371801e-01 -2.2328833e-01
     -2.8091627e-01  5.4120939e-02 -1.4878798e-02]
    [ 3.210856  3.2125535 4.8033743 3.3305957 2.6941013 1.9417268 2.006547
      1.9118829 1.9458938 1.853673  1.7844198 1.6339743 1.6456172 1.7596092
      1.8500254 1.9625275 2.0693698 2.3795512 2.5497463]

    hidden_layers=3, n_hidden = 128, epochs = 100, 
    #[0.04167414829134941, -0.02665860392153263, 0.0009721947135403752, 0.0012272428721189499, 0.06724070012569427, -0.06245207414031029, 0.22439619898796082, 0.0992811769247055, 0.14298135042190552, -0.044178254902362823, 0.29278919100761414, 0.10550514608621597, 0.4928971529006958, 0.21692661941051483, 0.5212540030479431, -0.01810707524418831, 0.2277292013168335, -0.20534707605838776, -0.13112092018127441]
    #[2.6064534187316895, 2.6872997283935547, 3.8904073238372803, 2.7063381671905518, 2.3648195266723633, 1.6723861694335938, 1.6408462524414062, 1.7071837186813354, 1.5176011323928833, 1.5497140884399414, 1.4597985744476318, 1.4514492750167847, 1.3762459754943848, 1.5295066833496094, 1.5241572856903076, 1.7533166408538818, 1.772041916847229, 2.0087342262268066, 2.167294979095459]
    '''
    
    mean = np.array([0.15938489139080048, 0.13437163829803467, -0.10685830563306808, -0.20305974781513214, -0.12149039655923843, 0.27110394835472107, -0.024669695645570755, 0.09971354156732559, 0.13893413543701172, 0.18008248507976532, 0.2863767147064209, -0.016983674839138985, 0.0484212301671505, -0.10031658411026001, -0.08973441272974014, 0.2652119994163513, 0.26683610677719116, 0.334799587726593, 0.2667008936405182])
    std  = np.array([2.705018997192383, 2.7658820152282715, 3.8888356685638428, 2.7597925662994385, 2.4937331676483154, 1.7385445833206177, 1.6824743747711182, 1.749420166015625, 1.5652599334716797, 1.597946286201477, 1.4666709899902344, 1.4113574028015137, 1.3387941122055054, 1.4966223239898682, 1.489140272140503, 1.7257601022720337, 1.7191941738128662, 2.100419759750366, 2.261826515197754])
    return mean, std

if __name__ == '__main__':
    n_hidden = 128
#    run_training(n_hidden=n_hidden)
#    mean_scaled, std_scaled = run_diff(n_hidden=n_hidden)
    mean, std = save_later()
    run_ploting(mean=mean, std=std)



