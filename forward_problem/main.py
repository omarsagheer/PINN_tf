from forward_problem.equations.heat_f_pinn import HeatFPINN

n_int = 256
n_sb = 64
n_tb = 64

pde = HeatFPINN(n_int, n_sb, n_tb)

pde.plot_training_points()

history = pde.training(optimizer='lbfgs', config={'num_iterations': 1500, 'tolerance': 1e-5})
# history = pde.training(optimizer='adam', config={'num_epochs': 2000, 'learning_rate': 1e-3})

pde.plot_train_loss(history)
pde.plotting_solution()