from ensemble_base_configuration import BaseProjectileMotionEnsembleConfiguration
from plotter_ensemble_configuration import ProjectileMotionEnsemblePositionalViewer


class ProjectileMotionEnsembleConfiguration(BaseProjectileMotionEnsembleConfiguration):

	def __init__(self):
		super().__init__()

	def update_save_directory(self, path_to_save_directory=None):
		self.verify_visual_settings()
		self._visual_settings.update_save_directory(
			path_to_save_directory=path_to_save_directory)
		if self.simulations is not None:
			for simulation in self._simulations:
				simulation.update_save_directory(
					path_to_save_directory=path_to_save_directory)

	def view_ensemble_of_projectile_trajectories(self, cmap="jet", figsize=None, is_save=False):
		plotter = ProjectileMotionEnsemblePositionalViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_ensemble_of_projectile_trajectories(
			ensemble=self,
			cmap=cmap,
			figsize=figsize,
			is_save=is_save)

##