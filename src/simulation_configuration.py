from simulation_base_configuration import BaseProjectileMotionSimulationConfiguration
from plotter_simulation_configuration import (
	ProjectileMotionSimulationPositionalViewer,
	ProjectileMotionSimulationTemporalViewer)

class ProjectileMotionSimulationConfiguration(BaseProjectileMotionSimulationConfiguration):

	def __init__(self):
		super().__init__()

	def view_trajectory_of_projectile(self, percentage_tangent_samples=5, is_show_ground_level=False, is_show_launch_site=False, is_show_peak=False, is_show_tangent_at_peak=False, is_show_t_at_peak=False, figsize=None, is_save=False):
		plotter = ProjectileMotionSimulationPositionalViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_trajectory_of_projectile(
			simulation=self,
			percentage_tangent_samples=percentage_tangent_samples,
			is_show_ground_level=is_show_ground_level,
			is_show_launch_site=is_show_launch_site,
			is_show_peak=is_show_peak,
			is_show_tangent_at_peak=is_show_tangent_at_peak,
			is_show_t_at_peak=is_show_t_at_peak,
			figsize=figsize,
			is_save=is_save)

	def view_velocity_components_of_projectile(self, is_show_t_at_peak=False, figsize=None, is_save=False):
		plotter = ProjectileMotionSimulationTemporalViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_velocity_components_of_projectile(
			simulation=self,
			is_show_t_at_peak=is_show_t_at_peak,
			figsize=figsize,
			is_save=is_save)
	
	def view_energies_of_projectile(self, is_show_t_at_peak=False, figsize=None, is_save=False):
		plotter = ProjectileMotionSimulationTemporalViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_energies_of_projectile(
			simulation=self,
			is_show_t_at_peak=is_show_t_at_peak,
			figsize=figsize,
			is_save=is_save)

	def view_hamiltonian_variance_of_projectile(self, is_show_t_at_peak=False, figsize=None, is_save=False):
		plotter = ProjectileMotionSimulationTemporalViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_hamiltonian_variance_of_projectile(
			simulation=self,
			is_show_t_at_peak=is_show_t_at_peak,
			figsize=figsize,
			is_save=is_save)

##