from forward_problem.forward_pinn import ForwardPINN
import tensorflow as tf
import numpy as np

class HeatFPINN(ForwardPINN):
    def __init__(self, n_int, n_sb, n_tb, **kwargs):
        super().__init__(n_int, n_sb, n_tb, [0, 1], [0, 1], **kwargs)


    def initial_condition(self, x):
        x = tf.cast(x, tf.float32)
        pi = tf.constant(np.pi, dtype=x.dtype)
        return -tf.sin(pi * x)


    def left_boundary_condition(self, t):
        return tf.zeros((t.shape[0], 1))

    def right_boundary_condition(self, t):
        return tf.zeros((t.shape[0], 1))

    def exact_solution(self, inputs):
        t, x = inputs[:, 0], inputs[:, 1]
        pi = tf.constant(np.pi, dtype=x.dtype)
        u = -tf.exp(-pi ** 2 * t) * tf.sin(pi * x)
        return u

    def compute_pde_residual(self, input_int):
        # Compute the solution and its gradients
        with tf.GradientTape() as tape2:
            tape2.watch(input_int)  # Watch inputs for second-order gradient
            with tf.GradientTape() as tape1:
                tape1.watch(input_int)  # Watch inputs for first-order gradient
                u = self.approximate_solution(input_int)  # Solution u(x, t)

            grad_u = tape1.gradient(u, input_int)  # Compute first-order gradient
            grad_u_t = grad_u[:, 0:1]  # Time derivative (u_t)
            grad_u_x = grad_u[:, 1:2]  # Spatial derivative (u_x)

        grad_u_xx = tape2.gradient(grad_u_x, input_int)[:, 1:2]  # Second spatial derivative (u_xx)

        # PDE residual: u_t - u_xx
        residual = grad_u_t - grad_u_xx
        return tf.reshape(residual, [-1, 1])  # Reshape residual for compatibility
