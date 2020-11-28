from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

class NLM:
    """The NLM class is essentially the FeedFoward class from in-class exercises and homeworks,
       the differences being:

        `.forward` method can return the feature map using the `return_feature_map` flag.

        `.perform_bayesian` method returns the posterior predictive samples, along with 
            the joint mean, joint variance, and posterior samples.

        `.get_prior_samples` method returns a model samples based on the priors.
    """
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

    def forward(self, weights, x, return_feature_map = False):
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

        if return_feature_map:
            return input[0,:,:].T # Transform into shape (n x p)

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

    def get_posterior_samples(self, x_matrix, y_matrix, x_test_matrix, prior_var, noise_var, samples):
        # Currently assumes 0 prior mean, need to change(?)
        '''Function to generate posterior predictive samples for Bayesian linear regression model'''
        prior_variance = np.diag(prior_var * np.ones(x_matrix.shape[1]))
        prior_precision = np.linalg.inv(prior_variance)

        joint_precision = prior_precision + x_matrix.T.dot(x_matrix) / noise_var
        joint_variance = np.linalg.inv(joint_precision)
        joint_mean = joint_variance.dot(x_matrix.T.dot(y_matrix)) / noise_var

        #sampling 100 points from the posterior
        posterior_samples = np.random.multivariate_normal(joint_mean.flatten(), joint_variance, size=samples)

        #take posterior predictive samples
        posterior_predictions = np.dot(posterior_samples, x_test_matrix.T)
        posterior_predictive_samples = posterior_predictions[np.newaxis, :, :] + np.random.normal(0, noise_var**0.5, size=(100, posterior_predictions.shape[0], posterior_predictions.shape[1]))
        posterior_predictive_samples = posterior_predictive_samples.reshape((100 * posterior_predictions.shape[0], posterior_predictions.shape[1]))

        return joint_mean, joint_variance, posterior_predictions, posterior_predictive_samples

    def perform_bayesian(self, x, y, x_test, prior_var=1, noise_var=0.3, samples=100):
        # compute feature map
        feature_map = self.forward(self.weights, x.reshape(1,-1), return_feature_map=True)
        feature_map_test = self.forward(self.weights, x_test.reshape(1,-1), return_feature_map=True)

        # add constant term to the feature map (for Bayesian regression)
        feature_map = np.hstack((np.ones((feature_map.shape[0],1)), feature_map))
        feature_map_test = np.hstack((np.ones((feature_map_test.shape[0],1)), feature_map_test))

        return self.get_posterior_samples(feature_map, y, feature_map_test, prior_var, noise_var, samples=samples)

    def get_prior_samples(self, x_test, prior_var=1, noise_var=0.3, num_models=100):
        feature_map_test = self.forward(self.weights, x_test.reshape(1,-1), return_feature_map=True)
        W_random = np.random.normal(loc=0, scale=prior_var**0.5, size=(num_models, self.params['H'][-1])) # CHECK WITH WP ABOUT SCALE THEY USED FOR PAPER

        return feature_map_test.dot(W_random.T) + np.random.normal(loc=0, scale=noise_var**0.5, size=(x_test.shape[0], num_models))
