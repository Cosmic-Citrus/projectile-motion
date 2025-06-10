import numpy as np
import matplotlib.pyplot as plt
from plotter_base_configuration import BasePlotterConfiguration
from label_mapping_configuration import LabelMappingConfiguration


class BaseProjectileMotionSimulationViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	def plot_legend(self, fig, ax, simulation, label_mapping):
		parameters = (
			"drag coefficient",
			"air density",
			"g-acceleration",
			"launch speed",
			"launch angle")
		sub_title_labels = list()
		for parameter in parameters:
			if (parameter == "g-acceleration") and simulation.environment_parameters["is g=g(z)"]:
				partial_label = simulation.environment_parameters["surface"]["labels"]["g-acceleration"]
			else:
				value = simulation.get_value(
					parameter=parameter)
				partial_label = label_mapping.get_full_label(
					parameter=parameter,
					value=value)
			sub_title_labels.append(
				partial_label)
		sub_title = ", ".join(
			sub_title_labels)
		t = simulation.get_value(
			parameter="t")
		dt = simulation.get_value(
			parameter="time step-size")
		number_time_steps = simulation.get_value(
			parameter="number time-steps")
		first_primary_title = label_mapping.get_time_label(
			t=t,
			dt=dt,
			number_time_steps=number_time_steps)
		primary_prefix = simulation.projectile.get_projectile_label()
		primary_title = "{}, {}".format(
			primary_prefix,
			first_primary_title)
		leg_title = label_mapping.get_space_corrected_label(
			top_label=primary_title,
			bottom_label=sub_title)
		# leg_title = "{}\n{}".format(
		# 	primary_title,
		# 	sub_title)
		handles, labels = ax.get_legend_handles_labels()
		leg = self.visual_settings.get_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			title=leg_title)
		return fig, ax, leg

class BaseProjectileMotionSimulationPositionalViewer(BaseProjectileMotionSimulationViewer):

	def __init__(self):
		super().__init__()

	@staticmethod
	def plot_ground_level(ax, facecolor="black", linestyle="-"):
		label = "Ground Level"
		z_at_ground = 0
		ax.axhline(
			y=z_at_ground,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def plot_launch_position(ax, simulation, facecolor="black", marker="d"):
		label = "Launch Site"
		(x, z) = simulation.launch_parameters["position"]
		ax.scatter(
			[x],
			[z],
			color=facecolor,
			marker=marker,
			label=label)
		return ax

	@staticmethod
	def plot_peak(ax, simulation, facecolor="black", marker="*", alpha=0.7):
		label = "Peak"
		x_at_peak = float(
			simulation.parameterization_at_peak["x"])
		z_at_peak = float(
			simulation.parameterization_at_peak["z"])
		ax.scatter(
			[x_at_peak],
			[z_at_peak],
			color=facecolor,
			marker=marker,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_tangent_at_peak(ax, simulation, percentage_tangent_samples, facecolor="black", linestyle="--", alpha=0.7):
		label = "Tangent at Peak"
		x_at_peak = float(
			simulation.parameterization_at_peak["x"])
		z_at_peak = float(
			simulation.parameterization_at_peak["z"])
		full_interval = simulation.x[-1] - simulation.x[0]
		tangent_interval = full_interval * percentage_tangent_samples / 100
		x_tangent = np.array([
			x_at_peak - tangent_interval / 2,
			x_at_peak + tangent_interval / 2])
		z_tangent = np.full(
			fill_value=z_at_peak,
			shape=x_tangent.size,
			dtype=float)
		ax.plot(
			x_tangent,
			z_tangent,
			color=facecolor,
			label=label,
			linestyle=linestyle,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_time_at_peak(ax, simulation, label_mapping):
		value = float(
			simulation.parameterization_at_peak["t"])
		label = label_mapping.get_full_label(
			parameter="t at peak",
			value=value)
		ax.scatter(
			list(),
			list(),
			label=label,
			color="none")
		return ax

	@staticmethod
	def plot_trajectory(ax, simulation, facecolor="darkorange", linestyle="-"):
		label = "Trajectory"
		ax.plot(
			simulation.x,
			simulation.z,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def get_save_name(is_show_ground_level, is_show_launch_site, is_show_peak, is_show_tangent_at_peak, is_show_t_at_peak, is_save):
		if is_save:
			base_name = "ProjectileMotionSimulation"
			s_ground = "wGround" if is_show_ground_level else "woGround"
			s_launch = "wLaunch" if is_show_launch_site else "woLaunch"
			s_peak = "wPeak" if is_show_peak else "woPeak"
			s_tangent = "wPeakTan" if is_show_tangent_at_peak else "woPeakTan"
			s_time = "wPeakTime" if is_show_t_at_peak else "woPeakTime"
			suffix = "-".join([
				s_ground,
				s_launch,
				s_peak,
				s_tangent,
				s_time])
			save_name = "{}-{}".format(
				base_name,
				suffix)
		else:
			save_name = None
		return save_name

	def autoformat_plot(self, ax, simulation):
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
		title = r"Projectile Motion"
		ax = self.visual_settings.autoformat_axis_labels(
			ax=ax,
			xlabel=xlabel,
			ylabel=ylabel,
			title=title)
		xlim = (
			simulation.x[0] - 1,
			simulation.x[-1] + 1)
		z_at_ground = 0
		smallest_z = np.min(
			simulation.z)
		largest_z = np.max(
			simulation.z)
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

class ProjectileMotionSimulationPositionalViewer(BaseProjectileMotionSimulationPositionalViewer):

	def __init__(self):
		super().__init__()

	def view_trajectory_of_projectile(self, simulation, percentage_tangent_samples=5, is_show_ground_level=False, is_show_launch_site=False, is_show_peak=False, is_show_tangent_at_peak=False, is_show_t_at_peak=False, figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		if not isinstance(percentage_tangent_samples, (int, float)):
			raise ValueError("invalid type(percentage_tangent_samples): {}".format(type(percentage_tangent_samples)))
		if (percentage_tangent_samples < 0) or (percentage_tangent_samples > 100):
			raise ValueError("invalid percentage_tangent_samples: {}".format(percentage_tangent_samples))
		if not isinstance(is_show_ground_level, bool):
			raise ValueError("invalid type(is_show_ground_level): {}".format(type(is_show_ground_level)))
		if not isinstance(is_show_launch_site, bool):
			raise ValueError("invalid type(is_show_launch_site): {}".format(type(is_show_launch_site)))
		if not isinstance(is_show_peak, bool):
			raise ValueError("invalid type(is_show_peak): {}".format(type(is_show_peak)))
		if not isinstance(is_show_tangent_at_peak, bool):
			raise ValueError("invalid type(is_show_tangent_at_peak): {}".format(type(is_show_tangent_at_peak)))
		if not isinstance(is_show_t_at_peak, bool):
			raise ValueError("invalid type(is_show_t_at_peak): {}".format(type(is_show_t_at_peak)))
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.plot_trajectory(
			ax=ax,
			simulation=simulation)
		if is_show_ground_level:
			ax = self.plot_ground_level(
				ax=ax)
		if is_show_launch_site:
			ax = self.plot_launch_position(
				ax=ax,
				simulation=simulation)
		if is_show_peak:
			ax = self.plot_peak(
				ax=ax,
				simulation=simulation)
		if is_show_tangent_at_peak:
			ax = self.plot_tangent_at_peak(
				ax=ax,
				simulation=simulation,
				percentage_tangent_samples=percentage_tangent_samples)
		if is_show_t_at_peak:
			ax = self.plot_time_at_peak(
				ax=ax,
				simulation=simulation,
				label_mapping=label_mapping)
		ax = self.autoformat_plot(
			ax=ax,
			simulation=simulation)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		save_name = self.get_save_name(
			is_show_ground_level=is_show_ground_level,
			is_show_launch_site=is_show_launch_site,
			is_show_peak=is_show_peak,
			is_show_tangent_at_peak=is_show_tangent_at_peak,
			is_show_t_at_peak=is_show_t_at_peak,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

class BaseProjectileMotionSimulationTemporalViewer(BaseProjectileMotionSimulationViewer):

	def __init__(self):
		super().__init__()

	@staticmethod
	def plot_speed_vs_time(ax, simulation, label_mapping, facecolor="steelblue", linestyle="-"):
		label = label_mapping.get_parameter_label(
			parameter="v")
		ax.plot(
			simulation.t,
			simulation.v,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def plot_analytic_terminal_speed_vs_time(ax, simulation, label_mapping, facecolor="crimson", linestyle="--"):
		label = label_mapping.get_parameter_label(
			parameter="analytic terminal speed")
		value = simulation.get_value(
			parameter="analytic terminal speed")
		if isinstance(value, (int, float)):
			ax.axhline(
				value,
				color=facecolor,
				linestyle=linestyle,
				label=label)
		else:
			ax.plot(
				simulation.t,
				value,
				color=facecolor,
				linestyle=linestyle,
				label=label)
		return ax

	@staticmethod
	def plot_dx_dt_vs_time(ax, simulation, label_mapping, facecolor="limegreen", linestyle="-"):
		label = label_mapping.get_parameter_label(
			parameter="dx/dt")
		ax.plot(
			simulation.t,
			simulation.dx_dt,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def plot_dz_dt_vs_time(ax, simulation, label_mapping, facecolor="darkorange", linestyle="-"):
		label = label_mapping.get_parameter_label(
			parameter="dz/dt")
		ax.plot(
			simulation.t,
			simulation.dz_dt,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def plot_potential_energy_vs_time(ax, simulation, label_mapping, facecolor="limegreen", linestyle="-", alpha=0.75):
		label = label_mapping.get_parameter_label(
			parameter="potential energy")
		ax.plot(
			simulation.t,
			simulation.potential_energy,
			color=facecolor,
			linestyle=linestyle,
			label=label,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_kinetic_energy_vs_time(ax, simulation, label_mapping, facecolor="steelblue", linestyle="-", alpha=0.75):
		label = label_mapping.get_parameter_label(
			parameter="kinetic energy")
		ax.plot(
			simulation.t,
			simulation.kinetic_energy,
			color=facecolor,
			linestyle=linestyle,
			label=label,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_lagrangian_vs_time(ax, simulation, label_mapping, facecolor="darkorange", linestyle="-", alpha=0.75):
		label = label_mapping.get_parameter_label(
			parameter="lagrangian")
		ax.plot(
			simulation.t,
			simulation.lagrangian,
			color=facecolor,
			linestyle=linestyle,
			label=label,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_hamiltonian_vs_time(ax, simulation, label_mapping, facecolor="black", linestyle="-", alpha=0.75):
		label = label_mapping.get_parameter_label(
			parameter="hamiltonian")
		ax.plot(
			simulation.t,
			simulation.hamiltonian,
			color=facecolor,
			linestyle=linestyle,
			label=label,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_hamiltonian_variance_vs_time(ax, simulation, label_mapping, facecolor="black", linestyle="-"):
		label = label_mapping.get_parameter_label(
			parameter="hamiltonian variance")
		ax.plot(
			simulation.t,
			simulation.hamiltonian_variance,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def plot_time_at_peak(ax, simulation, label_mapping, facecolor="black", linestyle=":"):
		value = simulation.get_value(
			"t at peak")
		label = label_mapping.get_full_label(
			parameter="t at peak",
			value=value)
		ax.axvline(
			value,
			color=facecolor,
			linestyle=linestyle,
			label=label)
		return ax

	@staticmethod
	def get_save_name(temporal_parameter, is_show_t_at_peak, is_save):
		if is_save:
			base_name = "ProjectileMotionSimulation-{}_VS_Time".format(
				temporal_parameter)
			s_time = "wPeakTime" if is_show_t_at_peak else "woPeakTime"
			save_name = "{}-{}".format(
				base_name,
				s_time)
		else:
			save_name = None
		return save_name

	def autoformat_plot(self, ax, ylabel, simulation, is_energy_variance):
		if is_energy_variance:
			y_major_fmt = None
		else:
			y_major_fmt = r"${:,.2f}$"
		ax = self.visual_settings.autoformat_axis_ticks_and_ticklabels(
			ax=ax,
			x_major_ticks=True,
			y_major_ticks=True,
			x_minor_ticks=True,
			y_minor_ticks=True,
			x_major_ticklabels=True,
			y_major_ticklabels=True,
			x_major_fmt=r"${:,.2f}$",
			y_major_fmt=y_major_fmt)
		ax = self.visual_settings.autoformat_grid(
			ax=ax,
			grid_color="gray")
		xlabel = r"Time [$s$]"
		title = r"Projectile Motion"
		ax = self.visual_settings.autoformat_axis_labels(
			ax=ax,
			xlabel=xlabel,
			ylabel=ylabel,
			title=title)
		xlim = (
			simulation.t[0],
			simulation.t[-1])
		ax = self.visual_settings.autoformat_axis_limits(
			ax=ax,
			xlim=xlim)
		return ax

class ProjectileMotionSimulationTemporalViewer(BaseProjectileMotionSimulationTemporalViewer):

	def __init__(self):
		super().__init__()

	def view_velocity_components_of_projectile(self, simulation, is_show_t_at_peak=False, figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		self.verify_visual_settings()
		if not isinstance(is_show_t_at_peak, bool):
			raise ValueError("invalid type(is_show_t_at_peak): {}".format(type(is_show_t_at_peak)))
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.plot_speed_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		if simulation.projectile.drag_coefficient > 0:
			ax = self.plot_analytic_terminal_speed_vs_time(
				ax=ax,
				simulation=simulation,
				label_mapping=label_mapping)
		ax = self.plot_dx_dt_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		ax = self.plot_dz_dt_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		if is_show_t_at_peak:
			ax = self.plot_time_at_peak(
				ax=ax,
				simulation=simulation,
				label_mapping=label_mapping)
		ylabel = r"Velocity [$\frac{m}{s}$]"
		ax = self.autoformat_plot(
			ax=ax,
			simulation=simulation,
			ylabel=ylabel,
			is_energy_variance=False)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		save_name = self.get_save_name(
			temporal_parameter="SpeedsAndVelocities",
			is_show_t_at_peak=is_show_t_at_peak,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

	def view_energies_of_projectile(self, simulation, is_show_t_at_peak=False, figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		self.verify_visual_settings()
		if not isinstance(is_show_t_at_peak, bool):
			raise ValueError("invalid type(is_show_t_at_peak): {}".format(type(is_show_t_at_peak)))
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.plot_potential_energy_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		ax = self.plot_kinetic_energy_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		ax = self.plot_lagrangian_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		ax = self.plot_hamiltonian_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		if is_show_t_at_peak:
			ax = self.plot_time_at_peak(
				ax=ax,
				simulation=simulation,
				label_mapping=label_mapping)
		ylabel = r"Energy [$J$]"
		ax = self.autoformat_plot(
			ax=ax,
			simulation=simulation,
			ylabel=ylabel,
			is_energy_variance=False)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		save_name = self.get_save_name(
			temporal_parameter="Energies",
			is_show_t_at_peak=is_show_t_at_peak,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

	def view_hamiltonian_variance_of_projectile(self, simulation, is_show_t_at_peak=False, figsize=None, is_save=False):
		label_mapping = LabelMappingConfiguration()
		self.verify_visual_settings()
		if not isinstance(is_show_t_at_peak, bool):
			raise ValueError("invalid type(is_show_t_at_peak): {}".format(type(is_show_t_at_peak)))
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.plot_hamiltonian_variance_vs_time(
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		if is_show_t_at_peak:
			ax = self.plot_time_at_peak(
				ax=ax,
				simulation=simulation,
				label_mapping=label_mapping)
		ylabel = r"Energy Variance [$\%$]"
		ax = self.autoformat_plot(
			ax=ax,
			simulation=simulation,
			ylabel=ylabel,
			is_energy_variance=True)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			simulation=simulation,
			label_mapping=label_mapping)
		save_name = self.get_save_name(
			temporal_parameter="EnergyVar",
			is_show_t_at_peak=is_show_t_at_peak,
			is_save=is_save)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

##