import numpy as np
from scipy.integrate import solve_ivp
from projectile_configuration import ProjectileSphereConfiguration
from plotter_base_configuration import BasePlotterConfiguration
from label_mapping_configuration import LabelMappingConfiguration


class BaseProjectileMotionEnvironmentConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._projectile = None
		self._environment_parameters = None
		self._launch_parameters = None
		self._time_step_parameters = None
		self._get_g_acceleration = None

	@property
	def projectile(self):
		return self._projectile
	
	@property
	def environment_parameters(self):
		return self._environment_parameters
	
	@property
	def launch_parameters(self):
		return self._launch_parameters

	@property
	def time_step_parameters(self):
		return self._time_step_parameters

	@property
	def get_g_acceleration(self):
		return self._get_g_acceleration
	
	def initialize_projectile(self, projectile):
		if not isinstance(projectile, ProjectileSphereConfiguration):
			raise ValueError("invalid type(projectile): {}".format(type(projectile)))
		self._projectile = projectile

	def initialize_environment(self, air_density, g_acceleration, mass_surface, radius_surface):
		if not isinstance(air_density, (int, float)):
			raise ValueError("invalid type(air_density): {}".format(type(air_density)))
		if air_density < 0:
			raise ValueError("air_density should be greater than or equal to zero")
		if g_acceleration is None:
			if not isinstance(mass_surface, (int, float)):
				raise ValueError("invalid type(mass_surface): {}".format(type(mass_surface)))
			if mass_surface <= 0:
				raise ValueError("invalid mass_surface: {}".format(mass_surface))
			if not isinstance(radius_surface, (int, float)):
				raise ValueError("invalid type(radius_surface): {}".format(type(radius_surface)))
			if radius_surface <= 0:
				raise ValueError("invalid radius_surface: {}".format(radius_surface))
			g_constant = 6.67e-11
			get_g_acceleration = lambda z : mass_surface * g_constant / np.square(z + radius_surface)
			# g_acceleration = get_g_acceleration(
			# 	z=0)
			is_g_depend_on_z = True
			fraction_label = r"$\frac{GM}{(R+z)^2}$"
			label_mapping = LabelMappingConfiguration()
			mass_value_label = label_mapping.get_condensed_scientific_notation_label(
				value=mass_surface)
			radius_value_label = label_mapping.get_condensed_scientific_notation_label(
				value=radius_surface)
			subscript_label = r"$M=${} $kg$, $R=${} $m$".format(
				mass_value_label,
				radius_value_label)
			g_unit_label = label_mapping.get_unit_label(
				parameter="g-acceleration")
			g_label = "$g$ $=$ {} {} ({})".format(
				fraction_label,
				g_unit_label,
				subscript_label)
		else:
			if mass_surface is not None:
				raise ValueError("invalid type(mass_surface): {}".format(mass_surface))
			if radius_surface is not None:
				raise ValueError("invalid type(radius_surface): {}".format(radius_surface))
			if not isinstance(g_acceleration, (int, float)):
				raise ValueError("invalid type(g_acceleration): {}".format(type(g_acceleration)))
			if g_acceleration <= 0:
				raise ValueError("g_acceleration should be greater than zero")
			get_g_acceleration = lambda z : g_acceleration
			is_g_depend_on_z = False
			g_label = None
		surface_parameters = {
			"mass" : mass_surface,
			"radius" : radius_surface,
			"labels" : {
				"g-acceleration" : g_label,
				},
			}
		environment_parameters = {
			"air density" : air_density,
			"g-acceleration" : g_acceleration,
			"is g=g(z)" : is_g_depend_on_z,
			"surface" : surface_parameters}
		self._environment_parameters = environment_parameters
		self._get_g_acceleration = get_g_acceleration

	def initialize_launch_parameters(self, launch_position, launch_speed, launch_angle):
		
		def get_autocorrected_launch_angle(launch_angle):
			if np.isinf(launch_angle) or np.isnan(launch_angle):
				raise ValueError("invalid launch_angle: {}".format(launch_angle))
			modified_launch_angle = float(
				launch_angle)
			two_pi = 2 * np.pi
			while modified_launch_angle < 0:
				modified_launch_angle += two_pi
			while modified_launch_angle > two_pi:
				modified_launch_angle -= two_pi
			# if (modified_launch_angle < 0) or (modified_launch_angle > np.pi):
			# 	raise ValueError("launch angle should be between 0 and pi")
			if (modified_launch_angle < 0) or (modified_launch_angle > np.pi / 2):
				## simpler to calculate range of trajectory
				raise ValueError("launch angle should be between 0 and pi/2")
			return modified_launch_angle

		if not isinstance(launch_position, (tuple, list, np.ndarray)):
			raise ValueError("invalid type(launch_position): {}".format(type(launch_position)))
		number_position_elements = len(
			launch_position)
		if number_position_elements != len(["x", "y"]):
			raise ValueError("invalid launch_position: {}".format(launch_position))
		if isinstance(launch_position, np.ndarray):
			size_of_positions_shape = len(
				launch_position.shape)
			if size_of_positions_shape != 1:
				raise ValueError("invalid launch_position.shape: {}".format(launch_position.shape))
		if launch_position[1] < 0:
			raise ValueError("launch_position should be at or above ground level")
		if not isinstance(launch_speed, (int, float)):
			raise ValueError("invalid type(launch_speed): {}".format(type(launch_speed)))
		if launch_speed <= 0:
			raise ValueError("launch speed should be greater than zero")
		if not isinstance(launch_angle, (int, float)):
			raise ValueError("invalid type(launch_angle): {}".format(type(launch_angle)))
		modified_launch_angle = get_autocorrected_launch_angle(
			launch_angle=launch_angle)
		launch_parameters = {
			"position" : launch_position,
			"speed" : launch_speed,
			"angle" : modified_launch_angle}
		self._launch_parameters = launch_parameters

	def initialize_time_step_parameters(self, number_time_steps, dt):
		if not isinstance(number_time_steps, int):
			raise ValueError("invalid type(number_time_steps): {}".format(type(number_time_steps)))
		if number_time_steps <= 0:
			raise ValueError("number_time_steps should be greater than zero")
		if not isinstance(dt, (int, float)):
			raise ValueError("invalid type(dt): {}".format(type(dt)))
		if dt <= 0:
			raise ValueError("dt should be greater than zero")
		t = np.arange(
			0,
			number_time_steps * dt + dt,
			dt)
		time_step_parameters = {
			"t" : t,
			"time step-size" : dt,
			"number time-steps" : number_time_steps}
		self._time_step_parameters = time_step_parameters

class BaseProjectileMotionKinematicsConfiguration(BaseProjectileMotionEnvironmentConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_speed(dx_dt, dz_dt):
		v = np.sqrt(
			np.square(dx_dt) + np.square(dz_dt))
		return v

	@staticmethod
	def get_lagrangian(kinetic_energy, potential_energy):
		lagrangian = kinetic_energy - potential_energy
		return lagrangian

	@staticmethod
	def get_hamiltonian(kinetic_energy, potential_energy):
		hamiltonian = kinetic_energy + potential_energy
		return hamiltonian

	@staticmethod
	def get_hamiltonian_variance(hamiltonian):
		hamiltonian_variance = np.abs(
			100 * (hamiltonian - hamiltonian[0]) / hamiltonian[0])
		return hamiltonian_variance

	def get_analytic_terminal_speed(self, t, z):
		if self.projectile.drag_coefficient == 0:
			v = np.inf
		else:
			numerator = 2 * self.projectile.mass * self.get_g_acceleration(z)
			denominator = self.environment_parameters["air density"] * self.projectile.drag_coefficient * self.projectile.cross_sectional_area
			v = np.sqrt(
				numerator / denominator)
		return v

	def get_potential_energy(self, z):
		potential_energy = self.projectile.mass * self.get_g_acceleration(z) * z
		return potential_energy

	def get_kinetic_energy(self, v):
		kinetic_energy = self.projectile.mass * np.square(v) / 2
		return kinetic_energy

	def get_adjusted_height_at_peak(self, z_at_peak):
		height = z_at_peak - self.launch_parameters["position"][1]
		return height

	def get_range_of_trajectory(self, x_at_ground):
		magnitude = x_at_ground - self.launch_parameters["position"][0]
		return magnitude

	def get_arc_length_of_trajectory(self, dx_dt, dz_dt):
		dt = self.time_step_parameters["time step-size"]
		arc_length = np.nansum(
			dt * np.sqrt(
				np.square(dx_dt) + np.square(dz_dt)))
		# arc_length = np.nansum(
		# 	dt * np.sqrt(
		# 		1 + np.square(dz_dt / dx_dt)))
		return arc_length

class BaseProjectileMotionSolverConfiguration(BaseProjectileMotionKinematicsConfiguration):

	def __init__(self):
		super().__init__()

	def solve_equations_of_motion(self, **ivp_kwargs):

		def get_derivatives(t, u, k):
			x, dx_dt, z, dz_dt = u
			g_acceleration = self.get_g_acceleration(
				z=z)
			v = np.sqrt(
				np.square(dx_dt) + np.square(dz_dt))
			d2x_dt2 = -1 * k / self.projectile.mass * v * dx_dt
			d2z_dt2 = -1 * k / self.projectile.mass * v * dz_dt - g_acceleration
			derivatives = [
				dx_dt,
				d2x_dt2,
				dz_dt,
				d2z_dt2]
			return derivatives

		def get_root_at_peak(t, u, k):
			return u[3]

		def get_root_at_ground(t, u, k):
			return u[2]

		def get_indices_at_nearest_values(arr, value):
			backward_index = None
			forward_index = None
			absolute_difference = np.abs(
				arr - value)
			index_at_nearest_value = np.argmin(
				absolute_difference)
			if arr[index_at_nearest_value] >= value:
				forward_index = index_at_nearest_value
				if index_at_nearest_value > 0:
					backward_index = index_at_nearest_value - 1
			else:
				backward_index = index_at_nearest_value
				if index_at_nearest_value < arr.size - 1:
					forward_index = index_at_nearest_value + 1
			return backward_index, forward_index

		get_root_at_ground.terminal = True ## end simulation at z=0
		get_root_at_ground.direction = -1 ## trigger root-finder(t) at z>0 --> z<0
		number_time_steps = int(
			self.time_step_parameters["number time-steps"])
		t = np.copy(
			self.time_step_parameters["t"])
		dt = float(
			self.time_step_parameters["time step-size"])
		phi = float(
			self.launch_parameters["angle"])
		x = float(
			self.launch_parameters["position"][0])
		z = float(
			self.launch_parameters["position"][1])
		dx_dt = np.cos(phi) * self.launch_parameters["speed"]
		dz_dt = np.sin(phi) * self.launch_parameters["speed"]
		u = tuple([
			x,
			dx_dt,
			z,
			dz_dt])
		k = np.prod([
			0.5,
			self.projectile.drag_coefficient,
			self.environment_parameters["air density"],
			self.projectile.cross_sectional_area])
		sol = solve_ivp(
			get_derivatives,
			t_span=(
				t[0],
				t[-1]),
			y0=u,
			t_eval=t,
			args=(
				k,),
			events=(
				get_root_at_peak,
				get_root_at_ground),
			**ivp_kwargs)
		x_at_ground = sol.y[0][-1]
		dx_dt_at_ground = sol.y[1][-1]
		z_at_ground = 0
		dz_dt_at_ground = sol.y[3][-1]
		dz_dt_at_peak = 0
		times_at_peak = sol.t_events[0]
		times_at_ground = sol.t_events[1]
		if times_at_peak.size == 0:
			raise ValueError("more time-steps are needed to reach the peak")
		if times_at_ground.size == 0:
			raise ValueError("more time-steps are needed to reach the ground")
		t_at_peak = times_at_peak[0]
		t_at_ground = times_at_ground[0]
		backward_index, forward_index = get_indices_at_nearest_values(
			arr=sol.t,
			value=t_at_peak)
		if (backward_index is None) or (forward_index is None):
			if (backward_index is None) and (forward_index is None):
				x_at_peak = np.nan
				dx_dt_at_peak = np.nan
				z_at_peak = np.nan
			elif backward_index is None:
				x_at_peak = sol.y[0][forward_index]
				dx_dt_at_peak = sol.y[1][forward_index]
				z_at_peak = sol.y[2][forward_index]
			else: # elif forward_index is None:
				x_at_peak = sol.y[0][backward_index]
				dx_dt_at_peak = sol.y[1][backward_index]
				z_at_peak = sol.y[2][backward_index]
		else:
			# t_before_peak = sol.t[backward_index]
			# t_after_peak = sol.t[forward_index]
			indices_about_time_at_peak = np.array([
				backward_index,
				forward_index])
			x_at_peak = np.mean(
				sol.y[0][indices_about_time_at_peak])
			dx_dt_at_peak = np.mean(
				sol.y[1][indices_about_time_at_peak])
			z_at_peak = np.max(
				sol.y[2])
		solution = {
			"t" : sol.t,
			"x" : sol.y[0],
			"dx/dt" : sol.y[1],
			"z" : sol.y[2],
			"dz/dt" : sol.y[3],
			"events" : {
				"peak" : {
					"t" : t_at_peak,
					"x" : x_at_peak,
					"dx/dt" : dx_dt_at_peak,
					"z" : z_at_peak,
					"dz/dt" : dz_dt_at_peak},
				"ground" : {
					"t" : t_at_ground,
					"x" : x_at_ground,
					"dx/dt" : dx_dt_at_ground,
					"z" : z_at_ground,
					"dz/dt" : dz_dt_at_ground},
				},
			}
		return solution

class BaseProjectileMotionSimulationConfiguration(BaseProjectileMotionSolverConfiguration):

	def __init__(self):
		super().__init__()
		self._solution = None
		self._t = None
		self._x = None
		self._z = None
		self._dx_dt = None
		self._dz_dt = None
		self._v = None
		self._analytic_terminal_speed = None
		self._potential_energy = None
		self._kinetic_energy = None
		self._lagrangian = None
		self._hamiltonian = None
		self._hamiltonian_variance = None
		self._t_at_peak = None
		self._adjusted_height_at_peak = None
		self._t_at_ground = None
		self._range_of_trajectory = None
		self._arc_length_of_trajectory = None
		self._parameterization_at_peak = None
		self._parameterization_at_ground = None

	@property
	def solution(self):
		return self._solution

	@property
	def t(self):
		return self._t

	@property
	def x(self):
		return self._x

	@property
	def z(self):
		return self._z

	@property
	def dx_dt(self):
		return self._dx_dt

	@property
	def dz_dt(self):
		return self._dz_dt

	@property
	def v(self):
		return self._v

	@property
	def analytic_terminal_speed(self):
		return self._analytic_terminal_speed

	@property
	def potential_energy(self):
		return self._potential_energy
	
	@property
	def kinetic_energy(self):
		return self._kinetic_energy

	@property
	def lagrangian(self):
		return self._lagrangian
	
	@property
	def hamiltonian(self):
		return self._hamiltonian

	@property
	def hamiltonian_variance(self):
		return self._hamiltonian_variance
	
	@property
	def t_at_peak(self):
		return self._t_at_peak
	
	@property
	def adjusted_height_at_peak(self):
		return self._adjusted_height_at_peak
	
	@property
	def t_at_ground(self):
		return self._t_at_ground
	
	@property
	def range_of_trajectory(self):
		return self._range_of_trajectory
	
	@property
	def arc_length_of_trajectory(self):
		return self._arc_length_of_trajectory

	@property
	def parameterization_at_peak(self):
		return self._parameterization_at_peak

	@property
	def parameterization_at_ground(self):
		return self._parameterization_at_ground

	def initialize_simulation(self, projectile, launch_position, launch_speed, launch_angle, number_time_steps, dt, air_density, g_acceleration, mass_surface, radius_surface):
		self.initialize_projectile(
			projectile=projectile)
		self.initialize_environment(
			air_density=air_density,
			g_acceleration=g_acceleration,
			mass_surface=mass_surface,
			radius_surface=radius_surface)
		self.initialize_launch_parameters(
			launch_position=launch_position,
			launch_speed=launch_speed,
			launch_angle=launch_angle)
		self.initialize_time_step_parameters(
			number_time_steps=number_time_steps,
			dt=dt)

	def run_simulation(self, **ivp_kwargs):
		solution = self.solve_equations_of_motion(
			**ivp_kwargs)
		events = solution.pop(
			"events")
		parameterization_at_peak = events.pop(
			"peak")
		parameterization_at_ground = events.pop(
			"ground")
		t_at_peak = parameterization_at_peak["t"]
		z_at_peak = parameterization_at_peak["z"]
		t_at_ground = parameterization_at_ground["t"]
		x_at_ground = parameterization_at_ground["x"]
		adjusted_height_at_peak = self.get_adjusted_height_at_peak(
			z_at_peak=z_at_peak)
		range_of_trajectory = self.get_range_of_trajectory(
			x_at_ground=x_at_ground)
		t = solution["t"]
		x = solution["x"]
		z = solution["z"]
		dx_dt = solution["dx/dt"]
		dz_dt = solution["dz/dt"]
		arc_length_of_trajectory = self.get_arc_length_of_trajectory(
			dx_dt=dx_dt,
			dz_dt=dz_dt)
		v = self.get_speed(
			dx_dt=dx_dt,
			dz_dt=dz_dt)
		analytic_terminal_speed = self.get_analytic_terminal_speed(
			t=t,
			z=z)
		potential_energy = self.get_potential_energy(
			z=z)
		kinetic_energy = self.get_kinetic_energy(
			v=v)
		lagrangian = self.get_lagrangian(
			kinetic_energy=kinetic_energy,
			potential_energy=potential_energy)
		hamiltonian = self.get_hamiltonian(
			kinetic_energy=kinetic_energy,
			potential_energy=potential_energy)
		hamiltonian_variance = self.get_hamiltonian_variance(
			hamiltonian=hamiltonian)
		number_time_steps = t.size
		self._solution = solution
		self._t = t
		self._x = x
		self._z = z
		self._dx_dt = dx_dt
		self._dz_dt = dz_dt
		self._v = v
		self._analytic_terminal_speed = analytic_terminal_speed
		self._potential_energy = potential_energy
		self._kinetic_energy = kinetic_energy
		self._lagrangian = lagrangian
		self._hamiltonian = hamiltonian
		self._hamiltonian_variance = hamiltonian_variance
		self._adjusted_height_at_peak = adjusted_height_at_peak
		self._range_of_trajectory = range_of_trajectory
		self._arc_length_of_trajectory = arc_length_of_trajectory
		self._parameterization_at_peak = parameterization_at_peak
		self._parameterization_at_ground = parameterization_at_ground
		self._time_step_parameters["number time-steps"] = number_time_steps

	def get_value(self, parameter):
		parameters = (
			"air density",
			"g-acceleration",
			"launch position",
			"launch speed",
			"launch angle",
			"t",
			"time step-size",
			"number time-steps",
			"analytic terminal speed",
			"v",
			"potential energy",
			"kinetic energy",
			"lagrangian",
			"hamiltonian",
			"hamiltonian variance",
			"t at peak",
			"z at peak",
			"adjusted height at peak",
			"t at ground",
			"x at ground",
			"range of trajectory",
			"arc-length of trajectory")
		if parameter in parameters:
			if parameter in ("air density", "g-acceleration"):
				key = parameter
				value = self.environment_parameters[key]
			elif parameter in ("launch position", "launch speed", "launch angle"):
				key = parameter.replace(
					"launch ",
					"")
				value = self.launch_parameters[key]
			elif parameter in ("t", "time step-size", "number time-steps"):
				key = parameter
				value = self.time_step_parameters[key]
			elif parameter in ("t at peak", "t at ground", "z at peak", "x at ground"):
				key = parameter[0]
				if "at peak" in parameter:
					value = self.parameterization_at_peak[key]
				else: # elif "at ground" in parameter:
					value = self.parameterization_at_ground[key]
			else:
				key = parameter.replace(
					" ",
					"_")
				key = key.replace(
					"-",
					"_")
				value = getattr(
					self,
					key)
			if value is None:
				modified_value = None
			elif isinstance(value, int):
				modified_value = int(
					value)
			elif isinstance(value, float):
				modified_value = float(
					value)
			elif isinstance(value, (tuple, list, np.ndarray)):
				if isinstance(value, np.ndarray):
					modified_value = np.array(
						value)
				else:
					f = type(value)
					modified_value = f(
						value)
			else:
				raise ValueError("invalid type(value): {}".format(type(value)))
		else:
			modified_value = self.projectile.get_value(
				parameter=parameter)
		return modified_value

##