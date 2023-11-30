from typing import Any, Tuple, List
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

def store_torch_model(model:Any, path:str)->None:
    '''
    Store torch model
    Input:
        @model: given model
        @path: path to store
    '''
    torch.save(model.state_dict(), path)

def load_torch_model(model:Any, path:str)->None:
    '''
    Load torch model
    Input:
        @model: given model
        @path: path to load
    '''
    model.load_state_dict(torch.load(path))
    return model

def store_dk_model(model:Any, path:str)->None:
    '''
    Store DK model
    Input:
        @model: given model
        @path: path to store
    '''
    store_torch_model(model, path)

def load_dk_model(model:Any, path:str)->None:
    '''
    Load DK model
    Input:
        @model: given model
        @path: path to load
    '''
    from src.model.dk import GPRegressionModel
    loaded_model = GPRegressionModel(model.train_inputs[0], model.train_targets, model.likelihood, model.feature_extractor, low_dim=True)
    return load_torch_model(loaded_model, path)