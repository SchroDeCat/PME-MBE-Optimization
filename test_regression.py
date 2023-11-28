from src import cross_validation, load_data_preprocess, train_dk
from src import GPRegressionModel, ExactGPRegressionModel, RFFGPRegressionModel, LargeFeatureExtractor
from typing import Any, Tuple
import gpytorch

CSV_URL = 'https://drive.google.com/uc?export=download&id=1K4GujltZ-7_YshNJwXMKEd4e0yTuu5q1'
CSV_NO_817_URL = 'https://drive.google.com/uc?export=download&id=19IgGl230AbBuaH8q9wOxZ2R4OtRfHb7c'

def test(gp_type:str, low_dim:int, noise_constraint:Any, output_scale_constraint:Any, train_iter:int=10, lr:float=1e-4, verbose:bool=True, spectrum_norm:bool=False,  **kwargs: Any)->Any:
    # Load the data
    train_x, X_test, train_y, y_test = load_data_preprocess(csv_url=CSV_NO_817_URL, X_attr=['T_cell', 'time','wavelength'], Y='RT', test_size=0.25, random_state=11)

    # Create the Gaussian process regression model
    data_dim = train_x.size(1)
    likelihoods = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
    if  gp_type == 'dk':
        feature_extractor = LargeFeatureExtractor(data_dim, low_dim, spectrum_norm=spectrum_norm)
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


    # Find optimal model hyperparameters
    model = train_dk(model, train_iter=train_iter, lr=lr, verbose=verbose, gp_type=gp_type)


    # Perform cross-validation
    train_loss, test_loss = cross_validation(model, train_x, train_y, verbose=verbose, lr=lr, train_iter=train_iter, gp_type=gp_type)

    print(f'Cross-validation mean squared error:\t test {test_loss} \t train: {train_loss}')


if __name__ == '__main__':
    test(gp_type='dk', low_dim=2, noise_constraint=None, output_scale_constraint=None, train_iter=10, lr=1e-2, verbose=True, spectrum_norm=False)