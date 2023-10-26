from typing import Any, Tuple
from torchmetrics import MeanSquaredLogError
import torch
import tqdm
import gpytorch

def posterior(model:Any, test_x: torch.tensor) -> Tuple[torch.tensor]:
    '''
    Generate posterior on given GP and list of candidates
    Input:
        @model: given GP
        @test_x: candidates to evaluate marginal posterior
    Return:
        @mean: posterior mean on test_x
        @variance: posterior variance on test_x
        @ucb: posterior ucb on test_x
        @lcb: posterior lcb on test_x
    '''
    model.model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model.model(test_x)
    return preds.mean, preds.variance, preds.confidence_region()[1], preds.confidence_region()[0]

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