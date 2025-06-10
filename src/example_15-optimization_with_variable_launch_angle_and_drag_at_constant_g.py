import numpy as np
from trajectory_optimization_configuration import TrajectoryOptimizationConfiguration


# is_save, path_to_save_directory = False, None
is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/projectile_motion/output/"


if __name__ == "__main__":

	## initialize parameters
	radius = 0.5
	mass = 25
	drag_coefficient_bounds = np.array([
		# 0,
		0.1,
		0.8])
	launch_position = (
		0,
		0)
	launch_speed = 120
	launch_angle_bounds = np.deg2rad(
		np.array([
			0,
			90]))
	air_density = 1.29
	g_acceleration = 9.8
	mass_surface = None
	radius_surface = None
	number_time_steps = 5000
	dt = 0.01

	## initialize optimization
	objective_quantities = (
		"longest t at peak",
		"longest z at peak",
		"longest adjusted height at peak",
		"longest t at ground",
		"longest x at ground",
		"longest range of trajectory",
		"longest arc-length of trajectory")
	for objective_quantity in objective_quantities:
		trajectory_optimizer = TrajectoryOptimizationConfiguration()
		trajectory_optimizer.initialize_visual_settings()
		trajectory_optimizer.update_save_directory(
			path_to_save_directory=path_to_save_directory)
		trajectory_optimizer.initialize_constants_and_optimization_variables(
			launch_angle=launch_angle_bounds,
			launch_speed=launch_speed,
			air_density=air_density,
			radius=radius,
			mass=mass,
			drag_coefficient=drag_coefficient_bounds,
			launch_position=launch_position,
			g_acceleration=g_acceleration,
			mass_surface=mass_surface,
			radius_surface=radius_surface,
			number_time_steps=number_time_steps,
			dt=dt)
		trajectory_optimizer.initialize_optimization(
			objective_quantity=objective_quantity,
			method="LSODA")
		trajectory_optimizer.initialize_ensemble(
			number_samples_per_variable=10,
			method="LSODA")

		## view
		trajectory_optimizer.view_optimization_parameter_space_by_two_variables(
			is_contour=True,
			figsize=(12, 7),
			is_save=is_save)
		trajectory_optimizer.view_optimization_parameter_space_by_two_variables(
			is_surface=True,
			figsize=(12, 7),
			is_save=is_save)

##