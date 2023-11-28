from typing import Any, Tuple, List
from torchmetrics import MeanSquaredLogError
import torch
import tqdm



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