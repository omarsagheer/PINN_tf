from abc import ABC

from forward_problem.forward_pinn_structure.base_f_pinn import BaseFPINN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class UtilFPINN(BaseFPINN, ABC):
    """Utility class for the forward PINN structure."""

    def get_points(self, n_points):
        inputs = self.convert(self.generate_sobol_points(n_points))
        output = self.approximate_solution(inputs).numpy().reshape(-1, )
        exact_output = self.exact_solution(inputs).numpy().reshape(-1, )
        return inputs, output, exact_output

    def relative_L2_error(self, n_points=10000):
        inputs, output, exact_output = self.get_points(n_points)
        err = (tf.reduce_mean((output - exact_output) ** 2) / tf.reduce_mean(exact_output ** 2)) ** 0.5
        print('L2 Relative Error Norm: {:.6e}'.format(err))
        return inputs, output, exact_output

    def plotting_solution(self, n_points=100000):
        inputs, output, exact_output = self.relative_L2_error(n_points)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1], inputs[:, 0], c=exact_output, cmap='jet')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('t')
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which='both', ls=':')
        im2 = axs[1].scatter(inputs[:, 1], inputs[:, 0], c=output, cmap='jet')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('t')
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which='both', ls=':')
        axs[0].set_title('Exact Solution')
        axs[1].set_title('Approximate Solution')

        plt.show()
        plt.close()

    def plot_training_points(self):
        # Plot the input training points
        input_sb_left_, _ = self.add_spatial_boundary_points_left()
        input_sb_right_, _ = self.add_spatial_boundary_points_right()
        input_tb_, _ = self.add_temporal_boundary_points()
        input_int_, _ = self.add_interior_points()

        plt.figure(figsize=(16, 8), dpi=150)
        plt.scatter(input_sb_left_[:, 1].numpy(), input_sb_left_[:, 0].numpy(), label='Left Boundary Points')
        plt.scatter(input_sb_right_[:, 1].numpy(), input_sb_right_[:, 0].numpy(), label='Right Boundary Points')
        plt.scatter(input_int_[:, 1].numpy(), input_int_[:, 0].numpy(), label='Interior Points')
        plt.scatter(input_tb_[:, 1].numpy(), input_tb_[:, 0].numpy(), label='Initial Points')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_train_loss(history):
        hist = history['total_loss']
        plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.xlabel("Iterations")
        plt.ylabel("Log10 Loss")
        plt.legend()
        plt.show()