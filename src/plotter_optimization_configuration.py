import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotter_base_configuration import BasePlotterConfiguration
from label_mapping_configuration import LabelMappingConfiguration


class BaseTrajectoryOptimizationViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_space_corrected_label(label):
		(top_label, bottom_label) = label.split(
			"\n")
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
		modified_label = "{}\n{}".format(
			top_label,
			bottom_label)
		return modified_label

	@staticmethod
	def get_objective_parameter(optimizer):
		objective_parameter = optimizer.objective_quantity.replace(
			"longest ",
			"")
		objective_parameter = objective_parameter.replace(
			"fastest ",
			"")
		return objective_parameter

	@staticmethod
	def get_save_name(optimizer, ensemble, is_contour, is_surface, is_save):
	
		def modify_string(s):
			modified_s = s.replace(
				"-",
				" ")
			modified_s = modified_s.title()
			modified_s = modified_s.replace(
				" ",
				"")
			return modified_s

		if is_save:
			objective_quantity = modify_string(
				optimizer.objective_quantity)
			variable_parameters = list()
			for variable_parameter in optimizer.variable_projectile_parameters + optimizer.variable_simulation_parameters:
				modified_parameter = modify_string(
					variable_parameter)
				variable_parameters.append(
					modified_parameter)
			combined_variable_parameters = "-".join(
				variable_parameters)
			save_name = "TrajectoryOptimization-{}_VS_{}_Var-{}".format(
				objective_quantity,
				optimizer.number_variables,
				combined_variable_parameters)
			if ensemble is not None:
				save_name += "-wEnsemble"
			if is_contour:
				save_name += "-wCntr"
			if is_surface:
				save_name += "-wSurf"
		else:
			save_name = None
		return save_name

	def get_mutually_common_label(self, optimizer, label_mapping, sep=", "):
		
		def get_label_by_projectile_identifiers(optimizer):
			if "radius" in optimizer.constant_projectile_parameters.keys():
				label = optimizer.optimal_simulation.projectile.get_projectile_label()
			else:
				label = None
			return label

		def get_label_by_projectile_parameters(optimizer, sep, label_mapping):
			parameters = (
				"mass",
				"drag coefficient")
			labels = list()
			for parameter in parameters:
				key = optimizer.convert_parameter_to_key(
					parameter=parameter)
				if (parameter in optimizer.constant_projectile_parameters.keys()) or (key in optimizer.constant_projectile_parameters.keys()):
					value = optimizer.optimal_simulation.get_value(
						parameter=parameter)
					partial_label = label_mapping.get_full_label(
						parameter=parameter,
						value=value)
					labels.append(
						partial_label)
			number_labels = len(
				labels)
			if number_labels == 0:
				label = None
			else:
				label = sep.join(
					labels)
			return label

		def get_label_by_simulation_parameters(optimizer, sep, label_mapping):
			parameters = (
				"launch angle",
				"launch speed",
				"air density",
				"g-acceleration")
			labels = list()
			for parameter in parameters:
				if (parameter == "g-acceleration") and optimizer.optimal_simulation.environment_parameters["is g=g(z)"]:
					partial_label = optimizer.optimal_simulation.environment_parameters["surface"]["labels"]["g-acceleration"]
				else:
					key = optimizer.convert_parameter_to_key(
						parameter=parameter)
					if (parameter in optimizer.constant_simulation_parameters.keys()) or (key in optimizer.constant_simulation_parameters.keys()):
						value = optimizer.optimal_simulation.get_value(
							parameter=parameter)
						partial_label = label_mapping.get_full_label(
							parameter=parameter,
							value=value)
					else:
						partial_label = None
				if partial_label is not None:
					labels.append(
						partial_label)
			number_labels = len(
				labels)
			if number_labels == 0:
				label = None
			else:
				label = sep.join(
					labels)
			return label

		def get_label_by_time_parameters(optimizer, label_mapping):
			t = optimizer.optimal_simulation.get_value(
				parameter="t")
			dt = optimizer.optimal_simulation.get_value(
				parameter="time step-size")
			number_time_steps = optimizer.optimal_simulation.get_value(
				parameter="number time-steps")
			label = label_mapping.get_time_label(
				t=t,
				dt=dt,
				number_time_steps=number_time_steps)
			return label

		projectile_identifiers_label = get_label_by_projectile_identifiers(
			optimizer=optimizer)
		projectile_parameters_label = get_label_by_projectile_parameters(
			optimizer=optimizer,
			sep=sep,
			label_mapping=label_mapping)
		simulation_parameters_label = get_label_by_simulation_parameters(
			optimizer=optimizer,
			sep=sep,
			label_mapping=label_mapping)
		time_label = get_label_by_time_parameters(
			optimizer=optimizer,
			label_mapping=label_mapping)
		top_labels, bottom_labels = list(), list()
		for label in (projectile_identifiers_label, projectile_parameters_label):
			if label is not None:
				top_labels.append(
					label)
		for label in (simulation_parameters_label, time_label):
			if label is not None:
				bottom_labels.append(
					label)
		top_label = ", ".join(
			top_labels)
		bottom_label = ", ".join(
			bottom_labels)
		label = label_mapping.get_space_corrected_label(
			top_label=top_label,
			bottom_label=bottom_label)
		return label

	def plot_legend(self, fig, ax, optimizer, label_mapping, title_suffix=None):
		handles, labels = ax.get_legend_handles_labels()
		leg_title = self.get_mutually_common_label(
			optimizer=optimizer,
			label_mapping=label_mapping)
		if title_suffix is not None:
			leg_title = "{}; {}".format(
				leg_title,
				title_suffix)
		leg_title = label_mapping.get_space_corrected_label(
			top_label=leg_title)
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
			**leg_kwargs)
		return fig, ax, leg

class BaseOneVariableTrajectoryOptimizationViewer(BaseTrajectoryOptimizationViewer):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_variable_parameter(optimizer):
		variable_parameters = optimizer.variable_projectile_parameters + optimizer.variable_simulation_parameters
		variable_parameter = variable_parameters[0]
		return variable_parameter

	def plot_objective_by_optimizer_ensemble(self, ax, optimizer, label_mapping, facecolor, alpha=0.75):
		variable_parameter = self.get_variable_parameter(
			optimizer=optimizer)
		objective_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		parameter_values = list()
		objective_values = list()
		for sim in optimizer.ensemble.simulations:
			parameter_value = sim.get_value(
				parameter=variable_parameter)
			objective_value = sim.get_value(
				parameter=objective_parameter)
			parameter_values.append(
				parameter_value)
			objective_values.append(
				objective_value)
		top_label = r"Bounded Ensemble (${:,}$ Simulations)".format(
			optimizer.ensemble.number_simulations)
		variable_symbol = label_mapping.get_parameter_label(
			parameter=variable_parameter)
		variable_units = label_mapping.get_unit_label(
			parameter=variable_parameter)
		if variable_units is None:
			bottom_label = r"${:,.2f}$ $≤$ {} ≤ ${:,.2f}$".format(
				optimizer.variable_bounds[0][0],
				variable_symbol,
				optimizer.variable_bounds[0][1])
		else:
			bottom_label = r"${:,.2f}$ {} $≤$ {} ≤ ${:,.2f}$ {}".format(
				optimizer.variable_bounds[0][0],
				variable_units,
				variable_symbol,
				optimizer.variable_bounds[0][1],
				variable_units)
		label = "{}\n{}".format(
			top_label,
			bottom_label)
		ax.plot(
			parameter_values,
			objective_values,
			color=facecolor,
			label=label,
			alpha=alpha)
		return ax

	def plot_objective_by_separate_ensemble(self, ax, optimizer, ensemble, label_mapping, facecolors, marker="o", alpha=0.8):
		variable_parameter = self.get_variable_parameter(
			optimizer=optimizer)
		objective_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		for sim, facecolor in zip(ensemble.simulations, facecolors):
			parameter_value = sim.get_value(
				parameter=variable_parameter)
			objective_value = sim.get_value(
				parameter=objective_parameter)
			label = label_mapping.get_full_label(
				parameter=variable_parameter,
				value=parameter_value)
			ax.scatter(
				[parameter_value],
				[objective_value],
				color=facecolor,
				label=label,
				marker=marker,
				alpha=alpha)
		return ax

	def plot_objective_by_optimal_simulation(self, ax, optimizer, label_mapping, facecolor, marker="*"):
		variable_parameter = self.get_variable_parameter(
			optimizer=optimizer)
		objective_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		parameter_value = optimizer.optimal_simulation.get_value(
			parameter=variable_parameter)
		objective_value = optimizer.optimal_simulation.get_value(
			parameter=objective_parameter)
		top_label = "Optimal Trajectory"
		middle_label = label_mapping.get_full_label(
			parameter=variable_parameter,
			value=parameter_value)
		bottom_label = label_mapping.get_full_label(
			parameter=objective_parameter,
			value=objective_value)
		label = "{}\n{}\n{}".format(
			top_label,
			middle_label,
			bottom_label)
		ax.scatter(
			[parameter_value],
			[objective_value],
			color=facecolor,
			marker=marker,
			label=label)
		return ax

	def autoformat_plot(self, ax, optimizer, ensemble, label_mapping):
		
		def update_limits(ax, optimizer, ensemble):
			variable_parameter = self.get_variable_parameter(
				optimizer=optimizer)
			objective_parameter = self.get_objective_parameter(
				optimizer=optimizer)
			parameter_values, objective_values = list(), list()
			if optimizer.ensemble is not None:
				for sim in optimizer.ensemble.simulations:
					parameter_value = sim.get_value(
						parameter=variable_parameter)
					objective_value = sim.get_value(
						parameter=objective_parameter)
					parameter_values.append(
						parameter_value)
					objective_values.append(
						objective_value)
			if ensemble is not None:
				for sim in ensemble.simulations:
					parameter_value = sim.get_value(
						parameter=variable_parameter)
					objective_value = sim.get_value(
						parameter=objective_parameter)
					parameter_values.append(
						parameter_value)
					objective_values.append(
						objective_value)
			optimal_parameter_value = optimizer.optimal_simulation.get_value(
				parameter=variable_parameter)
			optimal_objective_value = optimizer.optimal_simulation.get_value(
				parameter=objective_parameter)
			parameter_values.append(
				optimal_parameter_value)
			objective_values.append(
				optimal_objective_value)
			parameter_values = np.array(
				parameter_values)
			objective_values = np.array(
				objective_values)
			xlim = (
				np.min(
					parameter_values),
				np.max(
					parameter_values))
			ylim = (
				np.nanmin(
					objective_values),
				np.nanmax(
					objective_values))
			ax = self.visual_settings.autoformat_axis_limits(
				ax=ax,
				xlim=xlim,
				ylim=ylim)
			return ax

		def update_ticks(ax):
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
			return ax

		def update_labels(ax, optimizer, label_mapping):
			variable_parameter = self.get_variable_parameter(
				optimizer=optimizer)
			xlabel_suffix = label_mapping.get_full_label(
				parameter=variable_parameter,
				value=None)
			xlabel = "{}\n{}".format(
				variable_parameter.title(),
				xlabel_suffix)
			objective_parameter = self.get_objective_parameter(
				optimizer=optimizer)
			ylabel_suffix = label_mapping.get_full_label(
				parameter=objective_parameter,
				value=None)
			ylabel = "{}\n{}".format(
				objective_parameter.title(),
				ylabel_suffix)
			title = r"Projectile Motion Optimization"
			ax = self.visual_settings.autoformat_axis_labels(
				ax=ax,
				xlabel=xlabel,
				ylabel=ylabel,
				title=title)
			return ax

		ax = update_limits(
			ax=ax,
			optimizer=optimizer,
			ensemble=ensemble)
		ax = update_ticks(
			ax=ax)
		ax = update_labels(
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping)
		return ax

class OneVariableTrajectoryOptimizationViewer(BaseOneVariableTrajectoryOptimizationViewer):

	def __init__(self):
		super().__init__()

	def view_optimization(self, optimizer, ensemble=None, optimal_facecolor="darkorange", ensemble_cmap="jet", samples_facecolor="black", figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		fig, ax = plt.subplots(
			figsize=figsize)
		if optimizer.ensemble is not None:
			ax = self.plot_objective_by_optimizer_ensemble(
				ax=ax,
				optimizer=optimizer,
				label_mapping=label_mapping,
				facecolor=samples_facecolor)
		if ensemble is not None:
			(ensemble_facecolors, _) = self.visual_settings.get_rgb_facecolors(
				number_colors=ensemble.number_simulations,
				cmap=ensemble_cmap)
			ax = self.plot_objective_by_separate_ensemble(
				ax=ax,
				optimizer=optimizer,
				ensemble=ensemble,
				label_mapping=label_mapping,
				facecolors=ensemble_facecolors)
		ax = self.plot_objective_by_optimal_simulation(
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			facecolor=optimal_facecolor)
		ax = self.autoformat_plot(
			ax=ax,
			optimizer=optimizer,
			ensemble=ensemble,
			label_mapping=label_mapping)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			title_suffix=None)
		save_name = self.get_save_name(
			optimizer=optimizer,
			ensemble=ensemble,
			is_contour=False,
			is_surface=False,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name)

class BaseTwoVariableTrajectoryOptimizationViewer(BaseTrajectoryOptimizationViewer):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_variable_parameters(optimizer):
		variable_parameters = optimizer.variable_projectile_parameters + optimizer.variable_simulation_parameters
		(variable_x_parameter, variable_y_parameter) = variable_parameters
		return variable_x_parameter, variable_y_parameter

	@staticmethod
	def plot_3D_surface(ax, X, Y, Z, cmap, norm):
		surf_handle = ax.plot_surface(
			X,
			Y,
			Z,
			cmap=cmap,
			norm=norm)
		return ax, surf_handle

	def plot_2D_contour(self, ax, X, Y, Z, cmap, norm):
		contour_handle = ax.contour(
			X,
			Y,
			Z,
			colors="black",
			norm=norm)
		fmt = lambda x : r"${:,.3f}$".format(
			x)
		ax.clabel(
			contour_handle,
			inline=True,
			fmt=fmt,
			fontsize=self.visual_settings.tick_size)
		filled_contour_handle = ax.contourf(
			X,
			Y,
			Z,
			cmap=cmap,
			norm=norm)
		return ax, contour_handle, filled_contour_handle

	def plot_objective_by_optimal_simulation(self, ax, optimizer, label_mapping, optimal_facecolor, number_dimensions, marker="*"):
		variable_x_parameter, variable_y_parameter = self.get_variable_parameters(
			optimizer=optimizer)
		variable_z_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		(x_parameter_value, y_parameter_value) = optimizer.optimization_information["parameters"]
		z_parameter_value = optimizer.optimization_information["objective value"]
		top_label = "Optimal Trajectory"
		middle_label = label_mapping.get_full_label(
			parameter=variable_z_parameter,
			value=z_parameter_value)
		first_bottom_label = label_mapping.get_full_label(
			parameter=variable_x_parameter,
			value=x_parameter_value)
		second_bottom_label = label_mapping.get_full_label(
			parameter=variable_y_parameter,
			value=y_parameter_value)
		label = "{}\n{}\n{}, {}".format(
			top_label,
			middle_label,
			first_bottom_label,
			second_bottom_label)
		if number_dimensions == 2:
			coordinates = (
				[x_parameter_value],
				[y_parameter_value])
		elif number_dimensions == 3:
			coordinates = (
				[x_parameter_value],
				[y_parameter_value],
				[z_parameter_value])
		else:
			raise ValueError("invalid number_dimensions: {}".format(number_dimensions))
		optimal_handle = ax.scatter(
			*coordinates,
			color=optimal_facecolor,
			marker=marker,
			label=label)
		return ax, optimal_handle

	def get_xyz_grid(self, optimizer):

		def get_variable_samples(optimizer, parameter):
			key = optimizer.ensemble.convert_parameter_to_key(
				parameter=parameter)
			if parameter in optimizer.ensemble.variable_simulation_parameters.keys():
				arr = optimizer.ensemble.variable_simulation_parameters[parameter]
			else:
				arr = optimizer.ensemble.variable_projectile_parameters[key]
			return arr

		if optimizer.ensemble is None:
			raise ValueError("optimizer.ensemble is not initialized")
		variable_x_parameter, variable_y_parameter = self.get_variable_parameters(
			optimizer=optimizer)
		variable_z_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		x = get_variable_samples(
			optimizer=optimizer,
			parameter=variable_x_parameter)
		y = get_variable_samples(
			optimizer=optimizer,
			parameter=variable_y_parameter)
		X, Y = np.meshgrid(
			x,
			y,
			indexing="ij")
		z = list()
		for sim in optimizer.ensemble.simulations:
			z_at_sim = sim.get_value(
				parameter=variable_z_parameter)
			z.append(
				z_at_sim)
		z = np.array(
			z)

		Z = z.reshape((
			x.size,
			y.size))
		return (x, y, z), (X, Y, Z)

	def autoformat_plot(self, ax, optimizer, label_mapping, number_dimensions):
		
		def update_limits(ax, optimizer, number_dimensions):
			variable_x_parameter, variable_y_parameter = self.get_variable_parameters(
				optimizer=optimizer)
			objective_parameter = self.get_objective_parameter(
				optimizer=optimizer)
			x_parameter_values, y_parameter_values, objective_values = list(), list(), list()
			if optimizer.ensemble is not None:
				for sim in optimizer.ensemble.simulations:
					objective_value = sim.get_value(
						parameter=objective_parameter)
					for index_at_variable_dimension, parameter in enumerate([variable_x_parameter, variable_y_parameter]):
						parameter_value = sim.get_value(
							parameter=parameter)
						if index_at_variable_dimension == 0:
							x_parameter_values.append(
								parameter_value)
						else: # elif index_at_variable_dimension == 1:
							y_parameter_values.append(
								parameter_value)
					objective_values.append(
						objective_value)
			optimal_objective_value = optimizer.optimal_simulation.get_value(
				parameter=objective_parameter)
			objective_values.append(
				optimal_objective_value)
			x_parameter_values = np.array(
				x_parameter_values)
			y_parameter_values = np.array(
				y_parameter_values)
			objective_values = np.array(
				objective_values)
			xlim = (
				np.min(
					x_parameter_values),
				np.max(
					x_parameter_values))
			ylim = (
				np.min(
					y_parameter_values),
				np.max(
					y_parameter_values))
			if number_dimensions == 3:
				zlim = (
					np.nanmin(
						objective_values),
					np.nanmax(
						objective_values))
			else:
				zlim = None
			ax = self.visual_settings.autoformat_axis_limits(
				ax=ax,
				xlim=xlim,
				ylim=ylim,
				zlim=zlim)
			return ax

		def update_ticks(ax, number_dimensions):
			kwargs = dict()
			if number_dimensions == 3:
				kwargs["z_major_ticks"] = True
				kwargs["z_minor_ticks"] = True
				kwargs["z_major_ticklabels"] = True
				kwargs["z_minor_ticklabels"] = False
				kwargs["z_major_fmt"] = r"${:,.2f}$"
			ax = self.visual_settings.autoformat_axis_ticks_and_ticklabels(
				ax=ax,
				x_major_ticks=True,
				y_major_ticks=True,
				x_minor_ticks=True,
				y_minor_ticks=True,
				x_major_ticklabels=True,
				y_major_ticklabels=True,
				x_major_fmt=r"${:,.2f}$",
				y_major_fmt=r"${:,.2f}$",
				**kwargs)
			ax = self.visual_settings.autoformat_grid(
				ax=ax,
				grid_color="gray")
			return ax

		def update_labels(ax, optimizer, label_mapping, number_dimensions):
			variable_x_parameter, variable_y_parameter = self.get_variable_parameters(
				optimizer=optimizer)
			objective_parameter = self.get_objective_parameter(
				optimizer=optimizer)
			xlabel_suffix = label_mapping.get_full_label(
				parameter=variable_x_parameter,
				value=None)
			xlabel = "{}\n{}".format(
				variable_x_parameter.title(),
				xlabel_suffix)
			ylabel_suffix = label_mapping.get_full_label(
				parameter=variable_y_parameter,
				value=None)
			ylabel = "{}\n{}".format(
				variable_y_parameter.title(),
				ylabel_suffix)
			if number_dimensions == 3:
				zlabel_suffix = label_mapping.get_full_label(
					parameter=objective_parameter,
					value=None)
				zlabel = "{}\n{}".format(
					objective_parameter.title(),
					zlabel_suffix)
			else:
				zlabel = None
			title = r"Projectile Motion Optimization"
			ax = self.visual_settings.autoformat_axis_labels(
				ax=ax,
				xlabel=xlabel,
				ylabel=ylabel,
				zlabel=zlabel,
				title=title)
			return ax

		ax = update_limits(
			ax=ax,
			optimizer=optimizer,
			number_dimensions=number_dimensions)
		ax = update_ticks(
			ax=ax,
			number_dimensions=number_dimensions)
		ax = update_labels(
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			number_dimensions=number_dimensions)
		return ax

	def get_leg_title_suffix(self, optimizer, label_mapping):
		
		def get_variable_parameter_label(variable_parameter, bounds, label_mapping):
			(lower_bound, upper_bound) = bounds
			variable_symbol = label_mapping.get_parameter_label(
				parameter=variable_parameter)
			variable_units = label_mapping.get_unit_label(
				parameter=variable_parameter)
			if variable_units is None:
				label = r"${:,.2f}$ $≤$ {} ≤ ${:,.2f}$".format(
					lower_bound,
					variable_symbol,
					upper_bound)
			else:
				label = r"${:,.2f}$ {} $≤$ {} ≤ ${:,.2f}$ {}".format(
					lower_bound,
					variable_units,
					variable_symbol,
					upper_bound,
					variable_units)
			return label

		number_simulations_label = r"${:,}$ Simulations".format(
			optimizer.ensemble.number_simulations)
		variable_x_parameter, variable_y_parameter = self.get_variable_parameters(
			optimizer=optimizer)
		variable_suffixes = list()
		for variable_parameter, bounds in zip((variable_x_parameter, variable_y_parameter), optimizer.variable_bounds):
			variable_suffix = get_variable_parameter_label(
				variable_parameter=variable_parameter,
				bounds=bounds,
				label_mapping=label_mapping)
			variable_suffixes.append(
				variable_suffix)
		(x_parameter_label, y_parameter_label) = variable_suffixes
		suffix = "{} ({}, {})".format(
			number_simulations_label,
			x_parameter_label,
			y_parameter_label)
		return suffix

	def plot_color_bar(self, fig, ax, optimizer, handle_for_color_bar, label_mapping, **kwargs):
		objective_parameter = self.get_objective_parameter(
			optimizer=optimizer)
		top_title = objective_parameter.title()
		bottom_title = label_mapping.get_full_label(
			parameter=objective_parameter,
			value=None)
		title = "{}\n{}".format(
			top_title,
			bottom_title)
		cbar = self.visual_settings.get_color_bar(
			fig=fig,
			ax=ax,
			handle=handle_for_color_bar,
			title=title,
			**kwargs)
		return fig, ax, cbar

class TwoVariableTrajectoryOptimizationViewer(BaseTwoVariableTrajectoryOptimizationViewer):

	def __init__(self):
		super().__init__()

	def view_optimization(self, optimizer, optimal_facecolor="white", cmap="Oranges", is_contour=False, is_surface=False, figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		if not isinstance(is_contour, bool):
			raise ValueError("invalid type(is_contour): {}".format(type(is_contour)))
		if not isinstance(is_surface, bool):
			raise ValueError("invalid type(is_surface): {}".format(type(is_surface)))
		if not (is_contour or is_surface):
			raise ValueError("invalid combination of is_contour={} and is_surface={}".format(is_contour, is_surface))
		(x, y, z), (X, Y, Z) = self.get_xyz_grid(
			optimizer=optimizer)
		_, norm = self.visual_settings.get_rgb_facecolors(
			number_colors=int(
				np.ceil(
					np.nanmax(
						Z))),
			cmap=cmap)
		if is_surface:
			number_dimensions = 3
			fig = plt.figure(
				figsize=figsize)
			ax = fig.add_subplot(
				1,
				1,
				1,
				projection="3d")
			ax, surf_handle = self.plot_3D_surface(
				ax=ax,
				X=X,
				Y=Y,
				Z=Z,
				cmap=cmap,
				norm=norm)
			handle_for_color_bar = surf_handle
			if is_contour:
				raise ValueError("not yet implemented")
		else:
			number_dimensions = 2
			fig, ax = plt.subplots(
				figsize=figsize)
			ax, contour_handle, filled_contour_handle = self.plot_2D_contour(
				ax=ax,
				X=X,
				Y=Y,
				Z=Z,
				cmap=cmap,
				norm=norm)
			handle_for_color_bar = filled_contour_handle
		ax, optimal_handle = self.plot_objective_by_optimal_simulation(
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			number_dimensions=number_dimensions,
			optimal_facecolor=optimal_facecolor)
		ax = self.autoformat_plot(
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			number_dimensions=number_dimensions)
		leg_title_suffix = self.get_leg_title_suffix(
			optimizer=optimizer,
			label_mapping=label_mapping)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			optimizer=optimizer,
			label_mapping=label_mapping,
			title_suffix=leg_title_suffix)
		fig, ax, cbar = self.plot_color_bar(
			fig=fig,
			ax=ax,
			optimizer=optimizer,
			handle_for_color_bar=handle_for_color_bar,
			label_mapping=label_mapping,
			pad=0.1,
			shrink=0.675)
		save_name = self.get_save_name(
			optimizer=optimizer,
			ensemble=None,
			is_contour=is_contour,
			is_surface=is_surface,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name)

##