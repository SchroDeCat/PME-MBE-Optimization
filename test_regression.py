from src import cross_validation, load_data_preprocess, train_dk
from src import GPRegressionModel, ExactGPRegressionModel, RFFGPRegressionModel, LargeFeatureExtractor
from typing import Any, Tuple
import gpytorch
import torch

CSV_URL = 'https://drive.google.com/uc?export=download&id=1K4GujltZ-7_YshNJwXMKEd4e0yTuu5q1'
CSV_NO_817_URL = 'https://drive.google.com/uc?export=download&id=19IgGl230AbBuaH8q9wOxZ2R4OtRfHb7c'

def test(gp_type:str, low_dim:bool, X_attr, Y_attr, noise_constraint:Any, output_scale_constraint:Any, train_iter:int=10, lr:float=1e-4, verbose:bool=True, **kwargs: Any)->Any:
    # Load the data
    train_x, test_x, train_y, test_y = load_data_preprocess(csv_url=CSV_NO_817_URL, X_attr=X_attr, Y=Y_attr, test_size=0.25, random_state=11)

    # Find optimal model hyperparameters
    # model = train_dk(train_x=train_x, train_y=train_y, train_iter=train_iter, learning_rate=lr, verbose=verbose, gp_type=gp_type, noise_constraint=noise_constraint, output_scale_constraint=output_scale_constraint, low_dim=low_dim)

    # Perform cross-validation
    # train_loss, test_loss = cross_validation(train_x, train_y, verbose=verbose, learning_rate=lr, train_iter=train_iter, gp_type=gp_type, loss_type='l1', noise_constraint=noise_constraint, output_scale_constraint=output_scale_constraint,)
    # train_loss, test_loss = cross_validation(test_x, test_y, verbose=verbose, learning_rate=lr, train_iter=train_iter, gp_type=gp_type, loss_type='l1', noise_constraint=noise_constraint, output_scale_constraint=output_scale_constraint, low_dim=low_dim)
    train_loss, test_loss = cross_validation(torch.cat([train_x, test_x], dim=0), torch.cat([train_y, test_y]), verbose=verbose, learning_rate=lr, train_iter=train_iter, 
                                             gp_type=gp_type, loss_type='l1', noise_constraint=noise_constraint, output_scale_constraint=output_scale_constraint, kiss_gp=kwargs.get('kiss_gp', False), low_dim=low_dim,)

    print(f'Cross-validation mean squared error:\t test {test_loss} \t train: {train_loss}')


if __name__ == '__main__':
    # no noise constraint (observation noise = 0.6)
    # test(gp_type='dk', low_dim=True, X_attr=['T_cell', 'time','wavelength'], Y_attr='RT', noise_constraint=None, output_scale_constraint=None, train_iter=10, lr=1e-2, verbose=True, spectrum_norm=False)

    # strict noise constraint
    noise_constraint = gpytorch.constraints.Interval(1e-2, .1)
    # test(gp_type='gp_exact', low_dim=True, X_attr=['T_cell', 'time', 'wavelength','R_0'], Y_attr='RT', noise_constraint=noise_constraint, output_scale_constraint=None, train_iter=10, lr=1e-2, verbose=True, kiss_gp=True)
    # test(gp_type='dk', low_dim=True, X_attr=['T_cell', 'time', 'wavelength','R_0'], Y_attr='RT', noise_constraint=noise_constraint, output_scale_constraint=None, train_iter=100, lr=1e-2, verbose=True, kiss_gp=True)
    # test(gp_type='dk', low_dim=True, X_attr=['T_cell', 'time', 'wavelength','R_0'], Y_attr='RT', noise_constraint=noise_constraint, output_scale_constraint=None, train_iter=100, lr=1e-2, verbose=True, kiss_gp=False)
    test(gp_type='gp_exact', low_dim=True, X_attr=['T_cell', 'time', 'wavelength','R_0'], Y_attr='RT', noise_constraint=noise_constraint, output_scale_constraint=None, train_iter=100, lr=1e-2, verbose=True, kiss_gp=False)
    test(gp_type='gp_exact', low_dim=True, X_attr=['T_cell', 'time', 'wavelength','R_0'], Y_attr='RT', noise_constraint=noise_constraint, output_scale_constraint=None, train_iter=100, lr=1e-2, verbose=True, kiss_gp=True)