from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import SGLD, LargeFeatureExtractor, GPRegressionModel, ExactGPRegressionModel, RFFGPRegressionModel
from typing import Any, Tuple, List
import numpy as np
import pandas as pd
import gpytorch
import torch
import tqdm


def load_data_preprocess(csv_url:str, X_attr:Tuple[str], Y:str, **kwargs: Any)->Tuple[torch.tensor]:
    '''
    Load Data from speficied csv_url, preprocess data with standard scaler
    Input:
        @csv_url: url to csv file
        @X_attr: list of input attributes
        @Y: output attribute
    Return:
        X_train, X_test, y_train, y_test
    '''
    df = pd.read_csv(csv_url)
    _seed = kwargs.get('seed', 42)
    _test_size = kwargs.get('test_size', 0.25)
    _scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(df[X_attr], df[Y].to_numpy(), test_size=_test_size, random_state=_seed)

    X_train = _scaler.fit_transform(X_train)
    X_test = _scaler.transform(X_test)

    return torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

def train_seq_nn(model:torch.nn.Module, train_x:torch.tensor, train_y: torch.tensor, **kwargs: Any)->torch.nn.Sequential:

    """
    Train a sequential neural network
    Input:
        @model: given neural network architecture
        @train_x: train set input
        @train_y: train set label
    Return:
        @model: trained model
    """
    verbose = kwargs.get('verbose', False)
    loss_type = kwargs.get('loss_type', 'mse')
    train_iter = kwargs.get('train_iter', 10)
    lr = kwargs.get('learning_rate', 1e-6)
    data_size = train_x.size(0)
    batch_size = kwargs.get('batch_size', data_size)

    dataloader = torch.utils.data.DataLoader(list(zip(train_x, train_y.reshape([data_size, 1]))), batch_size=batch_size, shuffle=True)

    model.train()

    # optimizer = torch.optim.Adam(params=model.arc.parameters(), lr=lr)
    optimizer = torch.optim.SGD(params=model.arc.parameters(), lr=lr)
    
    # "Loss"
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    else:
        mps_device = torch.device('cpu')
    mps_device = torch.device('cpu')
    loss_type = loss_type.lower()
    if loss_type == "mse":
        loss_func = torch.nn.MSELoss().to(device=mps_device)
    elif loss_type == 'mlse':
        loss_func = MeanSquaredLogError().to(device=mps_device)
    elif loss_type== 'l1':
        loss_func = torch.nn.L1Loss().to(device=mps_device)
    else:
        raise NotImplementedError(f"{loss_type} not implemented")
    
    # training iterations
    iterator = tqdm.auto.tqdm(range(train_iter)) if verbose else range(train_iter)
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        loss_sum = 0
        for _x, _y in dataloader:
            # Get output from model
            output = model(_x)
            # Calc loss and backprop derivatives
            loss = loss_func(_y, output)
            loss.backward()
            optimizer.step()
            loss_sum += loss

        if verbose:
            iterator.set_postfix({"Loss": loss_sum.item() * batch_size / data_size})
    
    model.eval()
    return model

def train_dk(train_x:torch.tensor, train_y:torch.tensor, **kwargs: Any)->Any:
    '''
    Train a deep kernel GP
    input:
        @train_x: train set input
        @train_y: train set label
    Return:
        @model: trained model
    '''
    loss_type = kwargs.get('loss_type', 'nll')
    train_iter = kwargs.get('train_iter', 10)
    lr = kwargs.get('learning_rate', 1e-6)
    verbose = kwargs.get('verbose', False)
    low_dim = kwargs.get('low_dim', True)
    gp_type = kwargs.get('gp_type', 'dk')
    noise_constraint = kwargs.get('noise_constraint', None)
    output_scale_constraint = kwargs.get('output_scale_constraint', None)
    kiss_gp = kwargs.get('kiss_gp', False)

    # Create the Gaussian process regression model
    data_dim = train_x.size(1)
    likelihoods = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
    if  gp_type == 'dk':
        _spectrum_norm = kwargs.get('spectrum_norm', False)
        feature_extractor = LargeFeatureExtractor(data_dim, low_dim, spectrum_norm=_spectrum_norm)
        model = GPRegressionModel(
                        train_x=train_x,
                        train_y=train_y, 
                        gp_likelihood=likelihoods,
                        gp_feature_extractor=feature_extractor, 
                        low_dim=low_dim, 
                        output_scale_constraint=output_scale_constraint,
                    )
    elif gp_type == 'gp_exact':
        model = ExactGPRegressionModel(
                        train_x=train_x,
                        train_y=train_y, 
                        gp_likelihood=likelihoods,
                        low_dim=low_dim, 
                        output_scale_constraint=output_scale_constraint,
                        kiss_gp=kiss_gp,
                    )
    elif gp_type == 'rff':
        model = RFFGPRegressionModel(
                        train_x=train_x,
                        train_y=train_y, 
                        gp_likelihood=likelihoods,
                        low_dim=low_dim, 
                        output_scale_constraint=output_scale_constraint,
        )
    else:
        raise NotImplementedError(f'gp_type {gp_type} not implemented')

    model.train()
    model.likelihood.train()

    # optimizer
    if gp_type == 'dk':
        params = [
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ]
        # optimizer = SGLD(params, lr=lr)
        optimizer = torch.optim.Adam(params=params, lr=lr)
    else:
        params = [
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ]
        optimizer = torch.optim.Adam(params=params, lr=lr)
        
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    if loss_type.lower() == "nll":
        loss_func = lambda pred, y: -mll(pred, y)
    elif loss_type.lower() == "mse":
        tmp_loss = torch.nn.MSELoss()
        loss_func = lambda pred, y: tmp_loss(pred.mean, y)
    else:
        raise NotImplementedError(f"{loss_type} not implemented")
    
    # training iterations
    iterator = tqdm.tqdm(range(train_iter)) if verbose else range(train_iter)
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop derivatives
        loss = loss_func(output, model.train_targets)
        loss.backward()
        optimizer.step()

        if verbose:
            if gp_type == 'dk':
                # iterator.set_postfix({f"Training Loss {loss_type.lower()}": loss.item(), "noise": model.likelihood.noise.item(), 'lengthscale': model.covar_module.base_kernel.base_kernel.lengthscale.item()})
                _length_scale = model.covar_module.base_kernel.lengthscale.detach() if not model.kiss_gp else model.covar_module.base_kernel.base_kernel.base_kernel.lengthscale.detach()
                iterator.set_postfix({f"Training Loss {loss_type.lower()}": loss.item(), "noise": model.likelihood.noise.item(), 'lengthscale': _length_scale})
            elif gp_type == 'gp_exact':
                _length_scale = model.covar_module.base_kernel.lengthscale.detach() if not model.kiss_gp else model.covar_module.base_kernel.base_kernel.lengthscale.detach()
                iterator.set_postfix({f"Training Loss {loss_type.lower()}": loss.item(), "noise": model.likelihood.noise.item(), 'lengthscale': _length_scale})
            else:
                iterator.set_postfix({f"Training Loss {loss_type.lower()}": loss.item()})                

    return model

def cross_validation(train_x:torch.tensor, train_y:torch.tensor, **kwargs: Any)->Tuple[torch.tensor]:
    '''
    Cross validation on given model
    Input:
        @train_x: train set input
        @train_y: train set label
    Return:
        @train_loss: train loss
        @test_loss: test loss
    '''
    import numpy as np
    from sklearn.model_selection import KFold
    from torchmetrics import MeanSquaredLogError

    _seed = kwargs.get('seed', 42)
    _k = kwargs.get('k', 5)
    _train_loss_type = kwargs.get('train_loss_type', 'mlse')
    _loss_type = kwargs.get('loss_type', 'mse')
    _train_iter = kwargs.get('train_iter', 10)
    _lr = kwargs.get('learning_rate', 1e-6)
    _model_type = kwargs.get('model_type', 'gp')
    _batch_size = kwargs.get('batch_size', train_x.size(0))
    _verbose = kwargs.get('verbose', False)
    _kiss_gp = kwargs.get('kiss_gp', False)

    kf = KFold(n_splits=_k, shuffle=True, random_state=_seed)
    kf.get_n_splits(train_x)

    train_loss = []
    test_loss = []

    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        if _model_type.lower() == 'gp':
            gp_type = kwargs.get('gp_type', 'dk')
            noise_constraint = kwargs.get('noise_constraint', None)
            output_scale_constraint = kwargs.get('output_scale_constraint', None)
            model = train_dk(train_x=X_train, train_y=y_train, loss_type='nll', train_iter=_train_iter, learning_rate=_lr, gp_type=gp_type, verbose=_verbose, noise_constraint=noise_constraint, output_scale_constraint=output_scale_constraint, kiss_gp=_kiss_gp)

        elif _model_type.lower() == 'nn':
            model = kwargs.get('model', None)
            model = train_seq_nn(model=model, loss_type=_train_loss_type, train_x=X_train, train_y=y_train,  train_iter=_train_iter, learning_rate=_lr, batch_size=_batch_size, verbose=_verbose)

        else:
            raise NotImplementedError(f"{_model_type} not implemented")
        
        # evaluate
        model.eval()
        y_pred = model(X_test).mean
        y_train_pred = model(X_train).mean

        if _loss_type.lower() == 'mse':
            loss_func = torch.nn.MSELoss()
        elif _loss_type.lower() == 'mlse':
            loss_func = MeanSquaredLogError()
        elif _loss_type.lower() == 'l1':
            loss_func = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f"{_loss_type} not implemented")

        train_loss.append(loss_func(y_train_pred, y_train).item())
        test_loss.append(loss_func(y_pred, y_test).item())

    if _verbose:
        print(f"{_k}-Fold Cross validation train loss {_loss_type.lower()}: {train_loss}")
        print(f"{_k}-Fold Cross validation test loss {_loss_type.lower()}: {test_loss}")

    return torch.tensor(train_loss).mean(), torch.tensor(test_loss).mean()