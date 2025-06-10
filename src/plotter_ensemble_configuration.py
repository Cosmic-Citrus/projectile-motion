import numpy as np
import matplotlib.pyplot as plt
from plotter_base_configuration import BasePlotterConfiguration
from label_mapping_configuration import LabelMappingConfiguration


class BaseProjectileMotionEnsemblePositionalViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_mutually_common_label(ensemble, label_mapping, sep=", "):

		def get_label_by_categorized_parameters(ensemble, possible_parameters, constant_parameters, label_mapping, sep):
			first_simulation = ensemble.simulations[0]
			labels = list()
			for parameter in possible_parameters:
				if parameter in constant_parameters:
					if (parameter == "g-acceleration") and first_simulation.environment_parameters["is g=g(z)"]:
						partial_label = first_simulation.environment_parameters["surface"]["labels"]["g-acceleration"]
					else:
						value = first_simulation.get_value(
							parameter=parameter)
						partial_label = label_mapping.get_full_label(
							parameter=parameter,
							value=value)
					labels.append(
						partial_label)
			number_labels = len(
				labels)
			if number_labels == 0:
				parameters_label = None
			else:
				parameters_label = sep.join(
					labels)
			return parameters_label

		def get_label_by_time_parameters(ensemble, label_mapping):
			first_simulation = ensemble.simulations[0]
			dt = first_simulation.get_value(
				parameter="time step-size")
			# t = first_simulation.get_value(
			# 	parameter="t")
			# number_time_steps = first_simulation.get_value(
			# 	parameter="number time-steps")
			# time_label = label_mapping.get_time_label(
			# 	t=t,
			# 	dt=dt,
			# 	number_time_steps=number_time_steps)
			parameter_label = label_mapping.get_parameter_label(
				parameter="time step-size")
			unit_label = label_mapping.get_unit_label(
				parameter="time step-size")
			time_label = r"{} $=$ ${}$ {}".format(
				parameter_label,
				dt,
				unit_label)
			return time_label

		if "radius" in ensemble.constant_projectile_parameters.keys():
			first_simulation = ensemble.simulations[0]
			projectile_identifier_label = first_simulation.projectile.get_projectile_label()
		else:
			projectile_identifier_label = "Projectile Sphere"
		projectile_parameters_label = get_label_by_categorized_parameters(
			ensemble=ensemble,
			possible_parameters=(
				# "radius",
				"mass",
				"drag coefficient"),
			constant_parameters=list(
				ensemble.constant_projectile_parameters.keys()),
			label_mapping=label_mapping,
			sep=sep)
		simulation_parameters_label = get_label_by_categorized_parameters(
			ensemble=ensemble,
			possible_parameters=(
				"launch angle",
				"launch speed",
				"air density",
				"g-acceleration"),
			constant_parameters=list(
				ensemble.constant_simulation_parameters.keys()),
			label_mapping=label_mapping,
			sep=sep)
		time_label = get_label_by_time_parameters(
			ensemble=ensemble,
			label_mapping=label_mapping)
		top_labels = list()
		for label in (projectile_identifier_label, projectile_parameters_label):
			if label is not None:
				top_labels.append(
					label)
		bottom_labels = list()
		for label in (simulation_parameters_label, time_label):
			if label is not None:
				bottom_labels.append(
					label)
		top_label = ", ".join(
			top_labels)
		bottom_label = ", ".join(
			bottom_labels)
		number_characters_at_top_label = len(
			top_label)
		number_characters_at_bottom_label = len(
			bottom_label)
		if number_characters_at_top_label > number_characters_at_bottom_label:
			number_char_difference = number_characters_at_top_label - number_characters_at_bottom_label
			half_difference = number_char_difference // 2
			quarter_difference = half_difference // 2
			bottom_label = " "*quarter_difference + bottom_label
		elif number_characters_at_top_label < number_characters_at_bottom_label:
			number_char_difference = number_characters_at_bottom_label - number_characters_at_top_label
			half_difference = number_char_difference // 2
			quarter_difference = half_difference // 2
			top_label = " "*quarter_difference + top_label
		label = "{}\n{}".format(
			top_label,
			bottom_label)
		return label

	@staticmethod
	def get_exclusive_label(simulation, ensemble, label_mapping, sep="\n"):
		
		def get_label_by_categorized_parameters(simulation, ensemble, possible_parameters, label_mapping, sep):
			labels = list()
			variables = list(ensemble.variable_projectile_parameters.keys()) + list(ensemble.variable_simulation_parameters.keys())
			for parameter in possible_parameters:
				key = ensemble.convert_parameter_to_key(
					parameter=parameter)
				if (parameter in variables) or (key in variables):
					value = simulation.get_value(
						parameter=parameter)
					label = label_mapping.get_full_label(
						parameter=parameter,
						value=value)
					labels.append(
						label)
			number_labels = len(
				labels)
			if number_labels == 0:
				parameters_label = None
			else:
				parameters_label = sep.join(
					labels)
			return parameters_label

		projectile_parameters_label = get_label_by_categorized_parameters(
			simulation=simulation,
			ensemble=ensemble,
			possible_parameters=(
				"radius",
				"mass",
				"drag coefficient"),
			label_mapping=label_mapping,
			sep=sep)
		simulation_parameters_label = get_label_by_categorized_parameters(
			simulation=simulation,
			ensemble=ensemble,
			possible_parameters=(
				"launch angle",
				"launch speed",
				"air density",
				"g-acceleration"),
			label_mapping=label_mapping,
			sep=sep)
		labels = list()
		for label in (projectile_parameters_label, simulation_parameters_label):
			if label is not None:
				labels.append(
					label)
		number_labels = len(
			labels)
		if number_labels == 0:
			label = None
		else:
			label = sep.join(
				labels)
		return label

	@staticmethod
	def plot_trajectory(ax, simulation, label, facecolor, linestyle="-", alpha=0.8):
		ax.plot(
			simulation.x,
			simulation.z,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def get_save_name(ensemble, is_save):
		if is_save:
			variable_parameters = list()
			for variable_parameter in ensemble.variable_projectile_parameters.keys():
				modified_variable_parameter = ensemble.convert_parameter_to_key(
					variable_parameter)
				variable_parameters.append(
					modified_variable_parameter)
			for variable_parameter in ensemble.variable_simulation_parameters.keys():
				modified_variable_parameter = ensemble.convert_parameter_to_key(
					variable_parameter)
				variable_parameters.append(
					modified_variable_parameter)
			base_name = "ProjectileMotionEnsemble"
			number_variable_parameters = len(
				variable_parameters)
			if number_variable_parameters == 1:
				suffix = "-Var_{}".format(
					variable_parameters[0])
			else:
				variable_suffix = "-".join(
					variable_parameters)
				suffix = "-Var_{}".format(
					variable_suffix)
			save_name = "{}{}".format(
				base_name,
				suffix)			
		else:
			save_name = None
		return save_name

	def autoformat_plot(self, ax, ensemble):
		ax = self.visual_settings.autoformat_axis_ticks_and_ticklabels(
			ax=ax,
			x_major_ticks=True,
			y_major_ticks=True,
			x_minor_ticks=True,
			y_minor_ticks=True,
			x_major_ticklabels=True,
			y_major_ticklabels=True,
			x_major_fmt=r"${:,.2f}$",
			y_major_fmt=r"${:,.2f}$")
		ax = self.visual_settings.autoformat_grid(
			ax=ax,
			grid_color="gray")
		xlabel = "Length [$m$]\n(X-axis)"
		ylabel = "Height [$m$]\n(Z-axis)"
		title = r"Projectile Motion Ensemble"
		ax = self.visual_settings.autoformat_axis_labels(
			ax=ax,
			xlabel=xlabel,
			ylabel=ylabel,
			title=title)
		x, z = list(), list()
		for simulation in ensemble.simulations:
			x.append(
				np.min(
					simulation.x))
			x.append(
				np.max(
					simulation.x))
			z.append(
				np.min(
					simulation.z))
			z.append(
				np.max(
					simulation.z))
		smallest_x = min(
			x)
		largest_x = max(
			x)
		xlim = (
			smallest_x - 1,
			largest_x + 1)
		z_at_ground = 0
		smallest_z = min(
			z)
		largest_z = max(
			z)
		y_max = 1.05 * largest_z
		y_min = smallest_z - (y_max - largest_z)
		z_at_below_ground = z_at_ground - (y_max - largest_z)
		ylim = (
			z_at_below_ground,
			y_max)
		ax = self.visual_settings.autoformat_axis_limits(
			ax=ax,
			xlim=xlim,
			ylim=ylim)
		return ax

	def plot_legend(self, fig, ax, ensemble, label_mapping):
		handles, labels = ax.get_legend_handles_labels()
		leg_title = self.get_mutually_common_label(
			ensemble=ensemble,
			label_mapping=label_mapping)
		number_labels = len(
			labels)
		if number_labels <= 5:
			number_leg_columns = number_labels
		else:
			number_leg_columns = number_labels // 2
		leg_kwargs = dict()
		bbox_to_anchor = [
			0,
			-0.1375,
			1,
			1]
		leg_kwargs["bbox_to_anchor"] = bbox_to_anchor
		leg_kwargs["bbox_transform"] = fig.transFigure
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title,
			number_columns=number_leg_columns,
			**leg_kwargs)
		return fig, ax, leg

class ProjectileMotionEnsemblePositionalViewer(BaseProjectileMotionEnsemblePositionalViewer):

	def __init__(self):
		super().__init__()

	def view_ensemble_of_projectile_trajectories(self, ensemble, cmap, figsize=None, is_save=None):
		label_mapping = LabelMappingConfiguration()
		(rgb_facecolors, norm) = self.visual_settings.get_rgb_facecolors(
			number_colors=ensemble.number_simulations,
			cmap=cmap)
		fig, ax = plt.subplots(
			figsize=figsize)
		for simulation, rgb_facecolor in zip(ensemble.simulations, rgb_facecolors):
			label = self.get_exclusive_label(
				simulation=simulation,
				ensemble=ensemble,
				label_mapping=label_mapping)
			ax = self.plot_trajectory(
				ax=ax,
				simulation=simulation,
				label=label,
				facecolor=rgb_facecolor)
		ax = self.autoformat_plot(
			ax=ax,
			ensemble=ensemble)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			ensemble=ensemble,
			label_mapping=label_mapping)
		save_name = self.get_save_name(
			ensemble=ensemble,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

##