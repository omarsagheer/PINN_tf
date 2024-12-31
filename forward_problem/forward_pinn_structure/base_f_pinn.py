from abc import ABC, abstractmethod
import tensorflow as tf
import sobol_seq
from tensorflow import keras


class BaseFPINN(ABC):
    def __init__(self, n_int, n_sb, n_tb, time_domain=None, space_domain=None, lambda_u=10, n_hidden_layers=4, neurons=20, retrain_seed=42):

        if time_domain is None:
            time_domain = [0, 1]
        if space_domain is None:
            space_domain = [0, 1]

        self.n_int = n_int
        self.n_sb = n_sb
        self.n_tb = n_tb

        # Extrema of the solution domain (t,x)
        self.domain_extrema = tf.constant([time_domain, space_domain])

        # Parameter to balance the role of data and PDE
        self.lambda_u = lambda_u
        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = self.build_model(n_hidden_layers, neurons, retrain_seed)

        self.ms = lambda x: tf.reduce_mean(tf.square(x))
        self.log10 = lambda x: tf.math.log(x) / tf.math.log(tf.constant(10., dtype=x.dtype))

    # Function to linearly transform a tensor whose value is between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    @staticmethod
    def generate_sobol_points(n_points):
        return sobol_seq.i4_sobol_generate(2, n_points)

    @staticmethod
    def build_model(n_hidden_layers, neurons, retrain_seed):
        # Set seed for reproducibility
        tf.random.set_seed(retrain_seed)

        model = keras.Sequential()
        # 2 input features for time and space
        model.add(keras.layers.InputLayer(input_shape=(2,)))

        # Add hidden layers
        for _ in range(n_hidden_layers):
            model.add(keras.layers.Dense(neurons, activation='tanh'))

        # Output layer
        model.add(keras.layers.Dense(1))
        return model

    @abstractmethod
    def initial_condition(self, x):
        pass

    @abstractmethod
    def left_boundary_condition(self, t):
        pass

    @abstractmethod
    def right_boundary_condition(self, t):
        pass

    @abstractmethod
    def exact_solution(self, inputs):
        pass

    @abstractmethod
    def compute_pde_residual(self, input_int):
        pass

    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.generate_sobol_points(self.n_tb))
        input_tb = tf.tensor_scatter_nd_update(
            input_tb,
            tf.constant([[i, 0] for i in range(self.n_tb)]),
            tf.fill([self.n_tb], t0)
        )
        output_tb = tf.reshape(self.initial_condition(input_tb[:, 1]), [-1, 1])
        return input_tb, output_tb

    def add_spatial_boundary_points_left(self):
        x_left = self.domain_extrema[1, 0]
        input_sb = self.convert(self.generate_sobol_points(self.n_sb))
        input_sb = tf.tensor_scatter_nd_update(
            input_sb,
            tf.constant([[i, 1] for i in range(self.n_sb)]),
            tf.fill([self.n_sb], x_left)
        )
        output_sb_left = tf.reshape(self.left_boundary_condition(input_sb[:, 0]), [-1, 1])
        return input_sb, output_sb_left

    def add_spatial_boundary_points_right(self):
        x_right = self.domain_extrema[1, 1]
        input_sb = self.convert(self.generate_sobol_points(self.n_sb))
        input_sb = tf.tensor_scatter_nd_update(
            input_sb,
            tf.constant([[i, 1] for i in range(self.n_sb)]),
            tf.fill([self.n_sb], x_right)
        )
        output_sb_right = tf.reshape(self.right_boundary_condition(input_sb[:, 0]), [-1, 1])
        return input_sb, output_sb_right

        #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced

    def add_interior_points(self):
        input_int = self.convert(self.generate_sobol_points(self.n_int))
        output_int = tf.zeros([self.n_int, 1])
        return input_int, output_int

        # Function returning the training sets S_sb, S_tb, S_int as dataloader

    def assemble_datasets(self):
        input_sb_left, output_sb_left = self.add_spatial_boundary_points_left()
        input_sb_right, output_sb_right = self.add_spatial_boundary_points_right()
        input_tb, output_tb = self.add_temporal_boundary_points()
        input_int, output_int = self.add_interior_points()
        training_set_sb_left = tf.data.Dataset.from_tensor_slices((input_sb_left, output_sb_left)).batch(self.n_sb)
        training_set_sb_right = tf.data.Dataset.from_tensor_slices((input_sb_right, output_sb_right)).batch(self.n_sb)
        training_set_tb = tf.data.Dataset.from_tensor_slices((input_tb, output_tb)).batch(self.n_tb)
        training_set_int = tf.data.Dataset.from_tensor_slices((input_int, output_int)).batch(self.n_int)
        return training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int