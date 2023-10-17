import os
import pandas as pd
import numpy as np
import collections
from models.modelBase import modelBase

from models.utils import CHARAS_LIST

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from tqdm import tqdm
import faulthandler

faulthandler.enable()

MAX_EPOCH = 200
        

class CVAE_base(nn.Module, modelBase):
    def __init__(self, name, omit_char=[], device='cpu'):
        nn.Module.__init__(self)
        modelBase.__init__(self, name)
        self.beta_nn = None
        self.factor_nn = None
        
        self.optimizer = None
        self.criterion = None
        self.lambdas = None
        self.omit_char = omit_char
        
        self.factor_nn_pred = []
        
        self.device = device

        self.datashare_chara = pd.read_pickle('./data/datashare_re.pkl').astype(np.float64)
        self.p_charas = pd.read_pickle('./data/p_charas.pkl').astype(np.float64).reset_index()
        self.portfolio_ret=  pd.read_pickle('./data/portfolio_ret.pkl').astype(np.float64)
        self.mon_ret = pd.read_pickle('./data/month_ret.pkl').astype(np.float64)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self.pently_lambda = None
        self.repeat = 10
        self.lr = None
    
    def debug(self, month):
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][CHARAS_LIST]
        print(beta_nn_input)


    def _get_item(self, month):
        if month not in self.p_charas['DATE'].values:
            # find the closest month in p_charas to month
            month = self.p_charas['DATE'].values[np.argmin(np.abs(self.p_charas['DATE'].values - month))]
            
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][CHARAS_LIST] # (94, 94)
        labels = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST].T.values # (94, 1)
        beta_nn_input['ret-rf'] = labels
        align_df = beta_nn_input.copy(deep=False).dropna()
            
        factor_nn_input = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST]
         
        # exit(0) if there is any nan in align_df
        if align_df.isnull().values.any():
            assert False, f'There is nan in align_df of : {month}'
        # return stock index (L), beta_nn_input (94*94=P*N), factor_nn_input (94*1=P*1), labels (94, = N,)
        return align_df.index, align_df.values[:, :-1].T, factor_nn_input.T.values , align_df.values[:, -1].T
    
    
    def dataloader(self, period): 
        mon_list = pd.read_pickle('data/mon_list.pkl')
        mon_list = mon_list.loc[(mon_list >= period[0]) & (mon_list <= period[1])]
        beta_nn_input_set = []
        factor_nn_input_set = []
        label_set = []
        for mon in mon_list:
            _, _beta_input, _factor_input, label =  self._get_item(mon)
            beta_nn_input_set.append(_beta_input)
            factor_nn_input_set.append(_factor_input)
            label_set.append(label)
            
        beta_nn_input_set = torch.from_numpy(np.array(beta_nn_input_set)).float().to(self.device)
        factor_nn_input_set = torch.from_numpy(np.array(factor_nn_input_set)).float().to(self.device)
        label_set = torch.from_numpy(np.array(label_set)).float().to(self.device)

        dataset = TensorDataset(beta_nn_input_set, factor_nn_input_set, label_set)   
        return DataLoader(dataset, batch_size=1, shuffle=True)


    def forward(self, char, pfret):
        processed_char = self.beta_nn(char)
        processed_pfret, mu, log_var = self.factor_nn(pfret)
        return torch.sum(processed_char * processed_pfret, dim=1), mu, log_var

    
    # train_one_epoch
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)
            output, mu, log_var = self.forward(beta_nn_input, factor_nn_input)
            l1_regulization = torch.Tensor([torch.norm(i,p=1) for i in self.parameters()])
            # print(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0))

            loss = self.criterion(output, labels) + self.lambdas*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0) + \
                self.pently_lambda * torch.sum(l1_regulization)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_dataloader)


    def __valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_dataloader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output, mu, log_var = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels) + self.lambdas*torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.valid_dataloader)
    
    
    def train_model(self):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_dataloader = self.dataloader(self.train_period)
        self.valid_dataloader = self.dataloader(self.valid_period)
        self.test_dataloader = self.dataloader(self.test_period)
        
        train_info = collections.defaultdict(int)
        valid_info = collections.defaultdict(int)
        for times in range(self.repeat):
            # print(f'Start Training Model ({times})')
            min_error = np.Inf
            no_update_steps = 0
            valid_loss = []
            train_loss = []
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.reset_weight()
            pbar = tqdm(np.arange(MAX_EPOCH), desc ="Training")
            for i in pbar:
                # print(f'Epoch {i}')
                self.train()
                train_error = self.__train_one_epoch()
                train_loss.append(train_error)
                
                self.eval()
                # valid and early stop
                with torch.no_grad():
                    valid_error = self.__valid_one_epoch()
                pbar.set_postfix({f'train loss': np.mean(train_error), f'valid loss':np.mean(valid_error)})
                valid_loss.append(valid_error)
                if valid_error < min_error:
                    min_error = valid_error
                    no_update_steps = 0
                    # save model
                    torch.save(self.state_dict(), f'./saved_models/{times}{self.name}.pt')
                else:
                    no_update_steps += 1
                
                if no_update_steps > 5: # early stop, if consecutive 3 epoches no improvement on validation set
                    # print(f'Early stop at epoch {i}')
                    break
                # load from (best) saved model
                self.load_state_dict(torch.load(f'./saved_models/{times}{self.name}.pt'))
                
            train_info[times] = train_loss
            valid_info[times] = valid_loss
        self.load_ensemble_model()
        return train_info, valid_info

        
    def load_ensemble_model(self):
        params_list = collections.deque(maxlen=self.repeat)
        for num_model in range(self.repeat):
            # print(f'./saved_models/{num_model}{self.name}.pt')
            self.load_state_dict(torch.load(f'./saved_models/{num_model}{self.name}.pt'))
            params = self.state_dict()
            params_list.append(params)
        avg_params = self.average_params(params_list)        
        self.load_state_dict(avg_params)

    
        # ensemble estimation
    def average_params(self, params_list):
        assert isinstance(params_list, (tuple, list, collections.deque))
        n = len(params_list)
        if n == 1:
            return params_list[0]
        new_params = collections.OrderedDict()
        keys = None
        for i, params in enumerate(params_list):
            if keys is None:
                keys = params.keys()
            for k, v in params.items():
                if k not in keys:
                    raise ValueError('the %d-th model has different params'%i)
                if k not in new_params:
                    new_params[k] = v / n
                else:
                    new_params[k] += v / n
        return new_params
    

    def calBeta(self, month, skip_char=[]):
        _, beta_nn_input, _, _ = self._get_item(month) # beta input: 94*94 = P*N
        
        # if some variables need be omitted
        if len(skip_char):
            beta_nn_input = pd.DataFrame(beta_nn_input.T, columns=CHARAS_LIST) # N*P
            beta_nn_input[skip_char] = beta_nn_input[skip_char] * 0.0
            beta_nn_input = beta_nn_input.values.T # P*N
        
        beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device) # N*P
        return self.beta_nn(beta_nn_input) # N*K
    
    
    def calFactor(self, month, skip_char=[]):
        _, _, factor_nn_input, _ = self._get_item(month) # factor input: P*1
        
        # if some variables need be omitted
        if len(skip_char):
            factor_nn_input = pd.DataFrame(factor_nn_input.T, columns=CHARAS_LIST) # 1*P
            factor_nn_input[skip_char] = factor_nn_input[skip_char] * 0.0
            factor_nn_input = factor_nn_input.values.T # P*1

        factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device) # 1*P
        factor_pred,_,_ = self.factor_nn(factor_nn_input) # K*1
        
        self.factor_nn_pred.append(factor_pred.T)
        
        return factor_pred.T # K*1
    
    
    def inference(self, month):
        if len(self.omit_char) == 0:
            assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
            
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            
            assert mon_beta.shape[1] == mon_factor.shape[0], f"Dimension mismatch between mon_factor: {mon_factor.shape} and mon_beta: {mon_beta.shape}"
            
            # R_{N*1} = Beta_{N*K} @ F_{K*1}
            return mon_beta @ mon_factor
        else:
            ret_R = []
            for char in self.omit_char:
                mon_factor, mon_beta = self.calFactor(month, [char]), self.calBeta(month, [char])
                ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # N*1
                
            mon_factor, mon_beta = self.calFactor(month), self.calBeta(month)
            ret_R.append((mon_beta @ mon_factor).cpu().detach().numpy()) # also add complete result
            
            return np.array(ret_R).squeeze(2).T # N*m
    
    
    def cal_delayed_Factor(self, month):
        # calculate the last day of the previous month
        if self.refit_cnt == 0:
            avg_f_pred = self.factor_nn_pred[0] # input of the first predict take hat{f}_t
            # print(avg_f_pred.shape)
        else:
            avg_f_pred = torch.mean(torch.stack(self.factor_nn_pred[:self.refit_cnt]), dim=0)

        return avg_f_pred
    
    
    def reset_weight(self):
        for layer in self.beta_nn: # reset beta_nn parameters
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.factor_nn.children(): # reset factor_nn parameters
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                
        self.optimizer.state = collections.defaultdict(dict) # reset optimizer state


    def release_gpu(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.valid_dataloader is not None:
            del self.valid_dataloader
        if self.test_dataloader is not None:
            del self.test_dataloader
        torch.cpu.empty_cache()


class LinearVAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(LinearVAE, self).__init__()
        
        self.encoder = nn.Linear(94, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)
        self.decoder = nn.Linear(latent_dim,latent_dim)
    
    def encode(self, inputs):
        result = self.encoder(inputs)
        # result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  self.decoder(z), mu, log_var


class CVAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(CVAE, self).__init__()
        
        modules = []
        hidden_dims = [32, 16, 8]
        in_channels = 94
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = nn.Linear(latent_dim, latent_dim)
    
    def encode(self,inputs):
        result = self.encoder(inputs)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return  self.decoder(z), mu, log_var


class CVAE0(CVAE_base):
    def __init__(self, latent_dim, lambdas, lr=0.001, pently_lambda = 0.04, omit_char=[], device='cpu'):
        CVAE_base.__init__(self, name=f'CVAE0_{latent_dim}', omit_char=omit_char, device=device)
        # P -> K

        self.beta_nn = nn.Sequential(
            # output layer
            nn.Linear(94, latent_dim)
        )
        self.factor_nn = LinearVAE(latent_dim)

        self.lr = lr
        self.pently_lambda = pently_lambda
        self.criterion = nn.MSELoss().to(device)
        self.lambdas = lambdas
    


class CVAE1(CVAE_base):
    def __init__(self, latent_dim, lambdas, dropout=0.2, lr=0.001, pently_lambda = 0.04, omit_char=[], device='cpu'):
        CVAE_base.__init__(self, name=f'CVAE1_{latent_dim}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(32, latent_dim)
        )
        self.factor_nn = CVAE(latent_dim)
        
        self.lr = lr
        self.pently_lambda = pently_lambda
        self.criterion = nn.MSELoss().to(device)
        self.lambdas = lambdas
        

class CVAE2(CVAE_base):
    def __init__(self, latent_dim, lambdas, dropout=0.2, lr=0.001, pently_lambda = 0.04, omit_char=[], device='cpu'):
        CVAE_base.__init__(self, name=f'CVAE2_{latent_dim}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 2
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(16, latent_dim)
        )
        self.factor_nn = CVAE(latent_dim)
        
        self.lr = lr
        self.pently_lambda = pently_lambda
        self.criterion = nn.MSELoss().to(device)
        self.lambdas = lambdas
        

class CVAE3(CVAE_base):
    def __init__(self, latent_dim, lambdas, dropout=0.2, lr=0.001, pently_lambda = 0.04, omit_char=[], device='cpu'):
        CVAE_base.__init__(self, name=f'CVAE3_{latent_dim}', omit_char=omit_char, device=device)
        self.dropout = dropout
        # P -> 32 -> K
        self.beta_nn = nn.Sequential(
            # hidden layer 1
            nn.Linear(94, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 2
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # hidden layer 3
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # output layer
            nn.Linear(8, latent_dim)
        )
        self.factor_nn = CVAE(latent_dim)
        
        self.lr = lr
        self.pently_lambda = pently_lambda
        self.criterion = nn.MSELoss().to(device)
        self.lambdas = lambdas

# if __name__=='__main__':
#     latent_size = 3
#     lambdas = 1
#     pently_lambda = 0.01
#     model = CVAE1(latent_size, lambdas,lr=0.0001, pently_lambda=pently_lambda)
#     model.train_model()