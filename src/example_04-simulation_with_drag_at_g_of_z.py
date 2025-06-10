import numpy as np
from projectile_configuration import ProjectileSphereConfiguration
from simulation_configuration import ProjectileMotionSimulationConfiguration

	
# is_save, path_to_save_directory = False, None
is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/projectile_motion/output/"


if __name__ == "__main__":

	## initialize projectile
	radius = 0.5
	mass = 25
	drag_coefficient = 0.4
	projectile = ProjectileSphereConfiguration()
	projectile.initialize(
		radius=radius,
		mass=mass,
		drag_coefficient=drag_coefficient)

	## initialize simulation
	launch_position = (
		0,
		0)
	launch_speed = 120
	launch_angle = np.deg2rad(
		65)
	number_time_steps = 5000
	dt = 0.01
	air_density = 1.29
	g_acceleration = None
	mass_surface = 5.97e24
	radius_surface = 6.378e6
	projectile_motion = ProjectileMotionSimulationConfiguration()
	projectile_motion.initialize_visual_settings()
	projectile_motion.update_save_directory(
		path_to_save_directory=path_to_save_directory)
	projectile_motion.initialize_simulation(
		projectile=projectile,
		launch_position=launch_position,
		launch_speed=launch_speed,
		launch_angle=launch_angle,
		number_time_steps=number_time_steps,
		dt=dt,
		air_density=air_density,
		g_acceleration=g_acceleration,
		mass_surface=mass_surface,
		radius_surface=radius_surface)
	projectile_motion.run_simulation(
		method="LSODA")

	## view
	projectile_motion.view_trajectory_of_projectile(
		percentage_tangent_samples=12.5,
		is_show_ground_level=True,
		is_show_launch_site=True,
		is_show_peak=True,
		is_show_tangent_at_peak=True,
		is_show_t_at_peak=True,
		figsize=(12, 7),
		is_save=is_save)
	projectile_motion.view_velocity_components_of_projectile(
		is_show_t_at_peak=True,
		figsize=(12, 7),
		is_save=is_save)
	projectile_motion.view_energies_of_projectile(
		is_show_t_at_peak=True,
		figsize=(12, 7),
		is_save=is_save)
	projectile_motion.view_hamiltonian_variance_of_projectile(
		is_show_t_at_peak=True,
		figsize=(12, 7),
		is_save=is_save)

##