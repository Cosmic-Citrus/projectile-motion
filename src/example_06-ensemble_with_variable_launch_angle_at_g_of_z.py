import numpy as np
from ensemble_configuration import ProjectileMotionEnsembleConfiguration


# is_save, path_to_save_directory = False, None
is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/projectile_motion/output/"


if __name__ == "__main__":

	## initialize parameters
	radius = 0.5
	mass = 25
	drag_coefficient = 0
	launch_position = (
		0,
		0)
	launch_speed = 120
	variable_launch_angles = np.deg2rad(
		np.arange(
			10,
			91,
			5))
	number_time_steps = 5000
	dt = 0.01
	air_density = 1.29
	g_acceleration = None
	mass_surface = 5.97e24
	radius_surface = 6.378e6

	## initialize ensemble
	ensemble = ProjectileMotionEnsembleConfiguration()
	ensemble.initialize_visual_settings()
	ensemble.update_save_directory(
		path_to_save_directory=path_to_save_directory)
	ensemble.initialize_ensemble(
		radius=radius,
		mass=mass,
		drag_coefficient=drag_coefficient,
		number_time_steps=number_time_steps,
		dt=dt,
		launch_position=launch_position,
		launch_angle=variable_launch_angles,
		launch_speed=launch_speed,
		air_density=air_density,
		g_acceleration=g_acceleration,
		mass_surface=mass_surface,
		radius_surface=radius_surface,
		method="LSODA")

	## view
	ensemble.view_ensemble_of_projectile_trajectories(
		figsize=(12, 7),
		is_save=is_save)

##