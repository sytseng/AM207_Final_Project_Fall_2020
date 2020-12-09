from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam


class NeuralNet:
    """Implement a feed-forward neural network"""

    def __init__(self, architecture, random=None, weights=None):

        # Unpack model architecture
        self.params = {
            'dim_hidden': architecture['width'], # List of number of nodes per layer
            'dim_in': architecture['input_dim'],
            'dim_out': architecture['output_dim'],
            'activation_type': architecture['activation_fn_type'],
            'activation_params': architecture['activation_fn_params'],
        }
        self.activation = architecture['activation_fn']

        # Number of parameters (weights + biases) in input layer
        self.D_in = self.params['dim_in'] * self.params['dim_hidden'][0] + self.params['dim_hidden'][0]

        # Number of parameters (weights + biases) in hidden layers
        self.D_hidden = 0
        for i, h in enumerate(self.params['dim_hidden'][1:]):
            # Multiply previous layer width by current layer width plus the bias (current layer width)
            self.D_hidden += self.params['dim_hidden'][i] * h + h

        # Number of parameters (weights + biases) in output layer
        self.D_out = self.params['dim_hidden'][-1] * self.params['dim_out'] + self.params['dim_out']

        # Number of total parameters
        self.D = self.D_in + self.D_hidden + self.D_out

        # Set random state for reproducibility
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        # Initiate model parameters (weights + biases)
        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            assert weights.shape == (1, self.D)
            self.weights = weights

        # To inspect model training later
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x, return_features=False):
        """Forward pass given weights and input data.

        Returns:
            output (numpy.array): Model predictions of shape (n_mod, n_param, n_obs).
        """
        ''' Forward pass given weights and input '''
        H = self.params['dim_hidden']
        dim_in = self.params['dim_in']
        dim_out = self.params['dim_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == dim_in
            x = x.reshape((1, dim_in, -1))
        else:
            assert x.shape[1] == dim_in

        weights = weights.T # Shape: (n_obs, n_param)

        #input to first hidden layer
        W = weights[:H[0] * dim_in].T.reshape((-1, H[0], dim_in))
        b = weights[H[0] * dim_in:H[0] * dim_in + H[0]].T.reshape((-1, H[0], 1))
        input = self.activation(np.matmul(W, x) + b)
        index = H[0] * dim_in + H[0]

        assert input.shape[1] == H[0]

        #additional hidden layers
        for i in range(len(self.params['dim_hidden']) - 1):
            W = weights[index:index + H[i] * H[i+1]].T.reshape((-1, H[i+1], H[i]))
            index += H[i] * H[i+1]
            b = weights[index:index + H[i+1]].T.reshape((-1, H[i+1], 1))
            index += H[i+1]
            output = np.matmul(W, input) + b
            input = self.activation(output)

            assert input.shape[1] == H[i+1]

        # Return values from the last hidden layer if desired
        if return_features:
            return input # Shape: (n_mod, n_param, n_obs)

        #output layer
        W = weights[index:index + H[-1] * dim_out].T.reshape((-1, dim_out, H[-1]))
        b = weights[index + H[-1] * dim_out:].T.reshape((-1, dim_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['dim_out']

        return output


    def _make_objective(self, x_train, y_train, reg_param=0):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        def objective(W, t):
            squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
            mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
            return mean_error

        return objective, grad(objective)


    def _fit(self, objective, gradient, params):

        self.objective, self.gradient = (objective, gradient)

        ### set up optimization
        step_size = params.get('step_size', 0.01)
        max_iteration = params.get('max_iteration', 5000)
        check_point = params.get('check_point', 100)
        weights_init = params.get('init', self.weights.reshape((1, -1)))
        mass = params.get('mass', None)
        optimizer = params.get('optimizer', 'adam')
        random_restarts = params.get('random_restarts', 5)

        # Define callback function
        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} loss {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(weights, iteration))))
        call_back = params.get('call_back', call_back)

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


    def fit(self, x_train, y_train, params):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['dim_in']
        assert y_train.shape[0] == self.params['dim_out']

        # Make objective function for training
        reg_param = params.get('reg_param', 0) # No regularization by default
        objective, gradient = self._make_objective(x_train, y_train, reg_param)

        # Train model
        self._fit(objective, gradient, params)


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
        n_weights = self.params['dim_hidden'][-1] + 1 # Regression includes a bias term
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
        # noise = np.random.normal(loc=0, scale=noise_var**0.5, size=preds.shape)
        # preds = preds + noise

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
        n_weights = self.params['dim_hidden'][-1] + 1 # Regression includes a bias term
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
        # noise = np.random.normal(loc=0, scale=noise_var**0.5, size=preds.shape)
        # preds = preds + noise

        return preds


class LUNA(NLM):

    def __init__(self, architecture, random=None, weights=None):
        super(LUNA, self).__init__(architecture, random=random, weights=weights) # Inherit from NLM

        # Number of auxiliary functions to use
        self.params['M'] = architecture['auxiliary_functions']

        # Count number of parameters (weights + biases) in LUNA framework
        self.D_theta = self.D_in + self.D_hidden
        self.D_aux = self.params['M'] * self.D_out
        self.D = self.D_theta + self.D_aux # Total

        # Initiate model parameters (weights + biases)
        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            assert weights.shape == (1, self.D)
            self.weights = weights ## error?

        self.theta = self.weights[0][:self.D_theta]

        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x, return_features=False):
        """Forward pass for LUNA.

        Returns:
            output (numpy.array): Model predictions of shape (n_mod, n_param, n_obs).
        """

        assert weights.shape[1] == self.D

        # Obtain transformed features
        input = super(LUNA, self).forward(weights, x, return_features=True)

        # Return values from the last hidden layer if desired
        if return_features:
            return input # Shape: (n_mod, n_param, n_obs)

        # Set up for pass through output layer
        M = self.params['M'] # Number of auxiliary functions
        H = self.params['dim_hidden']
        dim_in = self.params['dim_in']
        dim_out = self.params['dim_out']
        index = self.D_theta
        weights = weights.T # Shape: (n_obs, n_param)

        # Pass through output layer
        W = weights[index:index + H[-1] * dim_out * M].T.reshape((M, dim_out, H[-1]))
        index += H[-1] * dim_out * M
        b = weights[index:].T.reshape((M, dim_out, 1))
        index += dim_out * M
        assert index == self.D
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['dim_out']

        return output


    def _finite_diff_grad(self, W_full, x_train):
        std_del = 0.1
#             delta_x = self.random.normal(0, std_del, size=(x_train.shape[1]) )  # use different perturbations for each x's
        delta_x = self.random.normal(0, std_del) # use one perturbation value for all x's
        x_for = x_train + delta_x
        x_return = self.forward(W_full, x_for) -  self.forward(W_full, x_train)

        return np.divide(x_return, delta_x) # M x dim_out x num of samples


    def _make_objective(self, x_train, y_train, reg_param=0, lambda_in=0):
        M = self.params['M']

        def objective(W_full, t):
            # Compute L_fit
            y_train_rep = np.tile(y_train, (M,1,1)) # repeat y_train with shape = dim_out x n_sample to M x dim_out x n_sample
            squared_error = np.linalg.norm(y_train_rep - self.forward(W_full, x_train), axis=1)**2
            L_fit = np.mean(squared_error) + reg_param * np.linalg.norm(W_full)

            # Comput L_diverse (#### Now only works for dim_in = 1 and dim_out = 1 ####)
            grad_FD = np.squeeze(self._finite_diff_grad(W_full, x_train)) # reshape to M x num of samples
            grad_angle = np.arctan(grad_FD) # compute the "angle of those gradients"
            grad_angle_rep = np.tile(grad_angle,(M,1,1)) # repeate the matrix to create M x M x num of samples
            grad_angle_rep_transpose = np.transpose(grad_angle_rep, axes = [1,0,2]) # transpose the M x M matrix so we can take pairwise differences between auxiliary functions
            coSimSqMat = np.mean(np.cos(grad_angle_rep - grad_angle_rep_transpose)**2, axis = -1) # take pairwise square cosine similarity between auxiliary functions, average over datapoints

#             norm_grad = grad_FD/(np.linalg.norm(grad_FD, axis = -1, keepdims=True)) # normalize along gradient of f wrt x to unit length
#             norm_grad = grad_FD/np.tile(np.linalg.norm(grad_FD, axis = -1),(x_train.shape[1],1)).T
#             coSimSqMat = np.matmul(norm_grad,norm_grad.T)**2 # matrix of all pairs of cosine similarity square

            coSimSq_uniq_pair = coSimSqMat[np.triu(np.ones((M,M),dtype=bool),k=1)] # taking the upper triagular part
            L_diverse = np.sum(coSimSq_uniq_pair)
#             L_diverse = np.sum(coSimSq_uniq_pair[np.random.choice(coSimSq_uniq_pair.shape[0],size = int(coSimSq_uniq_pair.shape[0]/3),replace=False )])

            return L_fit + lambda_in * L_diverse  # punish when coSimSq is large (close to 1)

        return objective, grad(objective)


    def fit(self, x_train, y_train, params):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['dim_in']
        assert y_train.shape[0] == self.params['dim_out']

        # Make objective function for training
        reg_param = params.get('reg_param', 0) # No regularization by default
        lambda_in = params.get('lambda_in', 0) # No diversity training by default
        objective, gradient = self._make_objective(x_train, y_train, reg_param, lambda_in)

        # Train model
        self._fit(objective, gradient, params)

        # Save feature map
        self.theta = self.weights[0][:self.D_theta]
