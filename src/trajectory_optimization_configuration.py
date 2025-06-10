from trajectory_optimization_base_configuration import BaseTrajectoryOptimizationConfiguration
from plotter_optimization_configuration import (
	OneVariableTrajectoryOptimizationViewer,
	TwoVariableTrajectoryOptimizationViewer)


class TrajectoryOptimizationConfiguration(BaseTrajectoryOptimizationConfiguration):

	def __init__(self):
		super().__init__()

	def view_optimization_parameter_space_by_one_variable(self, ensemble=None, optimal_facecolor="darkorange", ensemble_cmap="jet", samples_facecolor="black", figsize=None, is_save=False):
		plotter = OneVariableTrajectoryOptimizationViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_optimization(
			optimizer=self,
			ensemble=ensemble,
			optimal_facecolor=optimal_facecolor,
			ensemble_cmap=ensemble_cmap,
			samples_facecolor=samples_facecolor,
			figsize=figsize,
			is_save=is_save)

	def view_optimization_parameter_space_by_two_variables(self, optimal_facecolor="white", cmap="Oranges", is_contour=False, is_surface=False, figsize=None, is_save=False):
		plotter = TwoVariableTrajectoryOptimizationViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		plotter.view_optimization(
			optimizer=self,
			optimal_facecolor=optimal_facecolor,
			cmap=cmap,
			is_contour=is_contour,
			is_surface=is_surface,
			figsize=figsize,
			is_save=is_save)

##