from abc import ABC
import tensorflow as tf
from forward_problem.forward_pinn_structure.base_f_pinn import BaseFPINN
import numpy as np
import tensorflow_probability as tfp

class TrainingFPINN(BaseFPINN, ABC):
    """Training class for the forward PINN structure."""

    @tf.function
    def compute_loss(self, train_points):
        (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right,
         inp_train_tb, u_train_tb, inp_train_int) = train_points

        u_pred_sb_left = self.approximate_solution(inp_train_sb_left)
        u_pred_sb_right = self.approximate_solution(inp_train_sb_right)
        u_pred_tb = self.approximate_solution(inp_train_tb)

        assert u_pred_sb_left.shape == u_train_sb_left.shape
        assert u_pred_sb_right.shape == u_train_sb_right.shape
        assert u_pred_tb.shape == u_train_tb.shape

        # Compute the loss
        loss_sb_left = self.ms(u_pred_sb_left - u_train_sb_left)
        loss_sb_right = self.ms(u_pred_sb_right - u_train_sb_right)
        loss_tb = self.ms(u_pred_tb - u_train_tb)
        loss_int = self.ms(self.compute_pde_residual(inp_train_int))

        # boundary loss
        loss_u = loss_sb_left + loss_tb + loss_sb_right

        # Total loss with log scaling
        loss = self.log10(self.lambda_u * loss_u + loss_int)
        return loss, loss_u, loss_int


    def train_adam(self, num_epochs, learning_rate, verbose=True):
        """Train the PINN using Adam or similar gradient-based optimizers."""
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        history = {
            'total_loss': [],
            'boundary_loss': [],
            'pde_loss': [],
            'lr': []
        }
        training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int = self.assemble_datasets()

        for epoch in range(num_epochs):
            epoch_losses = []
            if verbose and epoch % max(1, num_epochs // 10) == 0:
                print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            with tf.GradientTape() as tape:
                for ((inp_train_sb_left, u_train_sb_left), (inp_train_sb_right, u_train_sb_right),
                        (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) \
                        in zip(training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int):
                    train_points = (inp_train_sb_left, u_train_sb_left, inp_train_sb_right, u_train_sb_right,
                                  inp_train_tb, u_train_tb, inp_train_int)
                    loss, loss_u, loss_int = self.compute_loss(train_points)
                    epoch_losses.append({
                        'total': loss,
                        'pde': self.log10(loss_int),
                        'boundary': self.log10(loss_u)
                    })
                    grads = tape.gradient(loss, self.approximate_solution.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.approximate_solution.trainable_variables))

            avg_losses = {
                k: np.mean([loss[k] for loss in epoch_losses]) for k in ['total', 'pde', 'boundary']
            }

            history['total_loss'].append(avg_losses['total'])
            history['boundary_loss'].append(avg_losses['boundary'])
            history['pde_loss'].append(avg_losses['pde'])
            history['lr'].append(optimizer.learning_rate.numpy())

            if verbose and epoch % max(1, num_epochs // 10) == 0:
                print(f"Total Loss: {avg_losses['total']:.6f} | "
                      f"Boundary Loss: {avg_losses['boundary']:.6f} | "
                      f"PDE Loss: {avg_losses['pde']:.6f} | "
                      f"LR: {optimizer.learning_rate.numpy():.6e}")

        if verbose:
            print(f"\nTraining completed. Final loss: {history['total_loss'][-1]:.6f}")

        return history

    def train_lbfgs(self, num_iterations, tolerance, verbose=True):
        """Train the PINN using L-BFGS optimizer."""
        history = {
            'total_loss': [],
            'boundary_loss': [],
            'pde_loss': [],
            'iter_count': 0,
        }
        # Get training data
        training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int = self.assemble_datasets()

        def flatten_list(lst):
            return tf.concat([tf.reshape(x, [-1]) for x in lst], axis=0)

        def unflatten_to_list_of_tensors(flat_tensor, shapes):
            result = []
            offset = 0
            for shape in shapes:
                size = tf.reduce_prod(shape)
                result.append(tf.reshape(flat_tensor[offset:offset + size], shape))
                offset += size
            return result

        # Get shapes and initial parameters
        shapes = [v.shape for v in self.approximate_solution.trainable_variables]
        init_params = flatten_list(self.approximate_solution.trainable_variables)

        def build_loss_and_grad_fn():
            def loss_value_and_gradients(params):
                # Update the model variables
                unflatten_params = unflatten_to_list_of_tensors(params, shapes)
                for var, value in zip(self.approximate_solution.trainable_variables, unflatten_params):
                    var.assign(value)

                # Get first batch from datasets
                train_points = next(zip(training_set_sb_left, training_set_sb_right, training_set_tb, training_set_int))
                inp_train_sb_left, u_train_sb_left = train_points[0]
                inp_train_sb_right, u_train_sb_right = train_points[1]
                inp_train_tb, u_train_tb = train_points[2]
                inp_train_int = train_points[3][0]

                with tf.GradientTape() as tape:
                    loss, loss_u, loss_int = self.compute_loss((inp_train_sb_left, u_train_sb_left,
                                                              inp_train_sb_right, u_train_sb_right,
                                                              inp_train_tb, u_train_tb,
                                                              inp_train_int))
                grads = tape.gradient(loss, self.approximate_solution.trainable_variables)


                if verbose and history['iter_count'] % max(1, num_iterations // 10) == 0:
                    print(f'\nIteration {history["iter_count"]}')
                    print(f"Loss: {loss:.6f} | Boundary Loss: {self.log10(loss_u):.6f} | "
                          f"PDE Loss: {self.log10(loss_int):.6f} | Iter: {history['iter_count']}")

                history['total_loss'].append(loss.numpy())
                history['boundary_loss'].append(self.log10(loss_u).numpy())
                history['pde_loss'].append(self.log10(loss_int).numpy())
                history['iter_count'] += 1
                return loss, flatten_list(grads)

            return loss_value_and_gradients

        if verbose:
            print("Starting L-BFGS optimization...")

        loss_and_grad_fn = build_loss_and_grad_fn()
        optimizer_results = tfp.optimizer.lbfgs_minimize(
            loss_and_grad_fn,
            initial_position=init_params,
            max_iterations=num_iterations,
            num_correction_pairs=50,
            tolerance=tolerance
        )

        # Update model with optimized parameters
        final_params = unflatten_to_list_of_tensors(optimizer_results.position, shapes)
        for var, value in zip(self.approximate_solution.trainable_variables, final_params):
            var.assign(value)

        if verbose:
            print(f"\nOptimization completed. Final loss: {optimizer_results.objective_value:.6f}")
            print(f"Converged: {optimizer_results.converged}")
            print(f"Number of iterations: {optimizer_results.num_iterations}")

        return history

    def training(self, optimizer='adam', config=None, verbose=True):
        """Main training function that delegates to the appropriate optimizer."""
        config = config or {}
        if optimizer == 'lbfgs':
            default_config = {
                'num_iterations': 100,
                'tolerance': 1e-5
            }
            num_iterations = config.get('num_iterations', default_config['num_iterations'])
            tolerance = config.get('tolerance', default_config['tolerance'])
            return self.train_lbfgs(num_iterations, tolerance, verbose)
        else:
            default_config = {
                'num_epochs': 1000,
                'learning_rate': 1e-3
            }
            num_epochs = config.get('num_epochs', default_config['num_epochs'])
            learning_rate = config.get('learning_rate', default_config['learning_rate'])
            return self.train_adam(num_epochs, learning_rate, verbose)