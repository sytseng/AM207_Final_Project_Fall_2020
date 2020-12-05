from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam


class NeuralNet:
    """Implement a feed-forward neural network"""

    def __init__(self, architecture, random=None, weights=None):
        # Sanity check
        assert len(architecture['width']) == architecture['hidden_layers']

        self.params = {'H': architecture['width'], # list of number of nodes per layer
                       'L': architecture['hidden_layers'], # number of hidden layers
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        # Input layer
        input_weights = architecture['input_dim'] * architecture['width'][0] + architecture['width'][0]

        # Loop over each layer
        hidden_weights = 0
        for i, h in enumerate(architecture['width'][1:]):
          # Multiply previous layer width by current layer width plus the bias (current layer width)
          hidden_weights += architecture['width'][i] * h + h

        # Output layer
        output_weights = (architecture['output_dim'] * architecture['width'][-1] + architecture['output_dim'])

        self.D = input_weights + hidden_weights + output_weights

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x, return_features=False):
        """Forward pass given weights and input data.

        Returns:
            output (numpy.array): Model predictions of shape (n_mod, n_param, n_obs).
        """
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        # Change
        #input to first hidden layer
        W = weights[:H[0] * D_in].T.reshape((-1, H[0], D_in))
        b = weights[H[0] * D_in:H[0] * D_in + H[0]].T.reshape((-1, H[0], 1))
        input = self.h(np.matmul(W, x) + b)
        index = H[0] * D_in + H[0]

        assert input.shape[1] == H[0]

        # Change
        #additional hidden layers
        for i in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H[i] * H[i+1]].T.reshape((-1, H[i+1], H[i]))
            index += H[i] * H[i+1]
            b = weights[index:index + H[i+1]].T.reshape((-1, H[i+1], 1))
            index += H[i+1]
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H[i+1]

        # Output values from the last hidden layer if desired
        if return_features:
            return input # Shape: (n_mod, n_param, n_obs)

        # Change
        #output layer
        W = weights[index:index + H[-1] * D_out].T.reshape((-1, D_out, H[-1]))
        b = weights[index + H[-1] * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']

        return output


    def make_objective(self, x_train, y_train, reg_param=None):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        if reg_param is None:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                sum_error = np.sum(squared_error)
                return sum_error

            return objective, grad(objective)

        else:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

            return objective, grad(objective)


    def fit(self, x_train, y_train, params, reg_param=None):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])

            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]


class NLM(NeuralNet):
    """Implement a neural linear model (NLM) for regression"""

    def _get_features(self, x, add_constant=False):
        """Help function for transforming data into features that can be used in
        Bayesian linear regression

        Args:
            x (numpy.array): Observed data of shape (n_param, n_obs)

        Returns:
            x_features (numpy.array): Transformed data of shape (n_param, n_obs)
        """

        x_features = self.forward(self.weights, x, return_features=True)
        x_features = x_features[0,:,:] # Reduces to shape (n_param, n_obs)
        if add_constant:
            x_features = np.vstack([np.ones((1, x_features.shape[1])), x_features])

        return x_features


    def get_prior_preds(self, x, w_prior_mean, w_prior_cov, noise_var=0.3, n_models=100):
        """Sample linear models from the given (normal) prior of weights
        and make predictions on the given data.

        Args:
            x (numpy.array): Data of shape (n_param, n_obs) to make predictions on.
            w_prior_mean (numpy.array): Means of regression model weights.
            w_prior_cov (numpy.array): Covariance matrix of regression model weights.
            noise_var (float): Variance of random noise in model prediction.
            n_models (int): Number of models to sample.

        Returns:
            preds (numpy.array): Predictions from sampled models; of shape (n_mod, n_obs).
        """

        # Handle single value inputs for w_mean and w_cov
        n_weights = self.params['H'][-1] + 1 # Regression includes a bias term
        w_prior_mean = np.array(w_prior_mean)
        w_prior_cov = np.array(w_prior_cov)
        if w_prior_mean.squeeze().shape == ():
            w_prior_mean = np.array([w_prior_mean] * n_weights)
        if w_prior_cov.squeeze().shape == ():
            w_prior_cov = np.eye(n_weights) * w_prior_cov

        # Transform data
        x_features = self._get_features(x, add_constant=True)

        # Sample models
        w_samples = np.random.multivariate_normal(
            mean=w_prior_mean,
            cov=w_prior_cov,
            size=n_models
        ) # Shape: (n_mod, n_param)

        # Make predictions
        preds = np.dot(w_samples, x_features) # Shape: (n_mod, n_obs)
        noise = np.random.normal(loc=0, scale=noise_var**0.5, size=preds.shape)
        preds = preds + noise

        return preds


    def get_posterior_preds(self, x, x_obs, y_obs, w_prior_cov, noise_var=0.3, n_models=100):
        """Infer the posterior from a normal prior with zero means, sample linear models
        from the posterior, and make predictions on the given data.

        Args:
            x (numpy.array): Data of shape (n_param, n_obs) to make predictions on.
            x_obs (numpy.array): Observed data of shape (n_param, n_obs).
            y_obs (numpy.array): Observed data of shape (1, n_obs).
            w_prior_cov (numpy.array): Covariance matrix of regression model weights.
            noise_var (float): Variance of random noise in model prediction.
            n_models (int): Number of models to sample.

        Returns:
            preds (numpy.array): Predictions from sampled models; of shape (n_mod, n_obs).
        """

        # Handle single value input for w_prior_cov
        n_weights = self.params['H'][-1] + 1 # Regression includes a bias term
        w_prior_cov = np.array(w_prior_cov)
        if w_prior_cov.squeeze().shape == ():
            w_prior_cov = np.eye(n_weights) * w_prior_cov

        # Transform data
        x_features = self._get_features(x, add_constant=True)
        x_obs_features = self._get_features(x_obs, add_constant=True)

        # Infer the posterior from the observed data and prior
        w_prior_precision = np.linalg.inv(w_prior_cov)
        w_post_precision = w_prior_precision + x_obs_features.dot(x_obs_features.T) / noise_var
        epsilon = 1e-5 # To ensure numerical stability of the matrix inverse operation
        w_post_cov = np.linalg.inv(w_post_precision + epsilon * np.ones(w_post_precision.shape[0]))
        w_post_mean = w_post_cov.dot(x_obs_features.dot(y_obs.flatten()) / noise_var) # Simplified because prior means are all zeros
        self.w_post_mean, self.w_post_cov = (w_post_mean, w_post_cov) # Save for later access

        # Sample models
        w_samples = np.random.multivariate_normal(
            mean=w_post_mean,
            cov=w_post_cov,
            size=n_models
        ) # Shape: (n_mod, n_param)

        # Make predictions
        preds = np.dot(w_samples, x_features) # Shape: (n_mod, n_obs)
        noise = np.random.normal(loc=0, scale=noise_var**0.5, size=preds.shape)
        preds = preds + noise

        return preds
