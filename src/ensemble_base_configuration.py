import itertools
import numpy as np
from projectile_configuration import ProjectileSphereConfiguration
from simulation_configuration import ProjectileMotionSimulationConfiguration
from plotter_base_configuration import BasePlotterConfiguration


class BaseProjectileMotionEnsembleParameterConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._number_projectile_variables = None
		self._number_simulation_variables = None
		self._number_variables = None
		self._number_simulations = None
		self._simulations = None
		self._constant_projectile_parameters = None
		self._variable_projectile_parameters = None
		self._constant_simulation_parameters = None
		self._variable_simulation_parameters = None

	@property
	def number_projectile_variables(self):
		return self._number_projectile_variables
	
	@property
	def number_simulation_variables(self):
		return self._number_simulation_variables
	
	@property
	def number_variables(self):
		return self._number_variables
	
	@property
	def number_simulations(self):
		return self._number_simulations
	
	@property
	def simulations(self):
		return self._simulations

	@property
	def constant_projectile_parameters(self):
		return self._constant_projectile_parameters
	
	@property
	def variable_projectile_parameters(self):
		return self._variable_projectile_parameters

	@property
	def constant_simulation_parameters(self):
		return self._constant_simulation_parameters
	
	@property
	def variable_simulation_parameters(self):
		return self._variable_simulation_parameters

	@staticmethod
	def convert_parameter_to_key(parameter):
		key = parameter.replace(
			"-",
			"_")
		key = key.replace(
			" ",
			"_")
		return key

	@staticmethod
	def get_base_constant_and_variable_parameters(parameter_names_and_values):
		constant_parameters, variable_parameters = dict(), dict()
		for parameter_name_and_value in parameter_names_and_values:
			(name, value) = parameter_name_and_value
			if value is None:
				constant_parameters[name] = value
			elif isinstance(value, (int, float, str)):
				constant_parameters[name] = value
			elif isinstance(value, (tuple, list, np.ndarray)):
				variable_parameters[name] = value
			else:
				raise ValueError("invalid type(value) at name={}: {}".format(name, type(value)))
		return constant_parameters, variable_parameters

	def initialize_projectile_parameters(self, radius, mass, drag_coefficient):
		
		def get_base_parameters(radius, mass, drag_coefficient):
			parameter_names_and_values = (
				("radius", radius),
				("mass", mass),
				("drag_coefficient", drag_coefficient))
			constant_projectile_parameters, variable_projectile_parameters = self.get_base_constant_and_variable_parameters(
				parameter_names_and_values=parameter_names_and_values)
			return constant_projectile_parameters, variable_projectile_parameters

		constant_projectile_parameters, variable_projectile_parameters = get_base_parameters(
			radius=radius,
			mass=mass,
			drag_coefficient=drag_coefficient)
		self._constant_projectile_parameters = constant_projectile_parameters
		self._variable_projectile_parameters = variable_projectile_parameters

	def initialize_simulation_parameters(self, number_time_steps, dt, launch_position, launch_angle, launch_speed, air_density, g_acceleration, mass_surface, radius_surface):

		def get_base_parameters(launch_angle, launch_speed, air_density, g_acceleration):
			parameter_names_and_values = (
				("launch angle", launch_angle),
				("launch speed", launch_speed),
				("air density", air_density),
				("g-acceleration", g_acceleration))
			constant_simulation_parameters, variable_simulation_parameters = self.get_base_constant_and_variable_parameters(
				parameter_names_and_values=parameter_names_and_values)
			return constant_simulation_parameters, variable_simulation_parameters

		def get_updated_constant_parameters(constant_simulation_parameters, number_time_steps, dt, launch_position, mass_surface, radius_surface):
			constant_simulation_parameters["number time-steps"] = number_time_steps
			constant_simulation_parameters["dt"] = dt
			constant_simulation_parameters["launch position"] = launch_position
			constant_simulation_parameters["mass surface"] = mass_surface
			constant_simulation_parameters["radius surface"] = radius_surface
			return constant_simulation_parameters

		constant_simulation_parameters, variable_simulation_parameters = get_base_parameters(
			launch_angle=launch_angle,
			launch_speed=launch_speed,
			air_density=air_density,
			g_acceleration=g_acceleration)
		constant_simulation_parameters = get_updated_constant_parameters(
			constant_simulation_parameters=constant_simulation_parameters,
			number_time_steps=number_time_steps,
			dt=dt,
			launch_position=launch_position,
			mass_surface=mass_surface,
			radius_surface=radius_surface)
		self._constant_simulation_parameters = constant_simulation_parameters
		self._variable_simulation_parameters = variable_simulation_parameters

	def initialize_variable_numbers(self):
		number_projectile_variables = len(
			self.variable_projectile_parameters.keys())
		number_simulation_variables = len(
			self.variable_simulation_parameters.keys())
		number_variables = number_projectile_variables + number_simulation_variables
		if number_variables == 0:
			raise ValueError("at least one variable parameter is required for the ensemble")
		self._number_projectile_variables = number_projectile_variables
		self._number_simulation_variables = number_simulation_variables
		self._number_variables = number_variables

	def initialize_simulations(self, **ivp_kwargs):

		def get_variable_projectile_parameter_value_combinations():
			if self.number_projectile_variables > 0:
				variable_value_combinations = list(
					itertools.product(
						*list(
							self.variable_projectile_parameters.values())))
			else:
				variable_value_combinations = None
			return variable_value_combinations

		def get_variable_simulation_parameter_value_combinations():
			if self.number_simulation_variables > 0:
				variable_value_combinations = list(
					itertools.product(
						*list(
							self.variable_simulation_parameters.values())))
			else:
				variable_value_combinations = None
			return variable_value_combinations

		def get_kwargs(variable_value_combination, constant_parameters, variable_names):
			kwargs = dict()
			for parameter, constant_value in constant_parameters.items():
				key = self.convert_parameter_to_key(
					parameter=parameter)
				kwargs[key] = constant_value
			if variable_value_combination is not None:
				for index_at_parameter, variable_value in enumerate(variable_value_combination):
					parameter = variable_names[index_at_parameter]
					key = self.convert_parameter_to_key(
						parameter=parameter)
					kwargs[key] = variable_value
			return kwargs

		def get_simulation(variable_projectile_parameter_value_combination=None, variable_simulation_parameter_value_combination=None, **ivp_kwargs):
			projectile_kwargs = get_kwargs(
				variable_value_combination=variable_projectile_parameter_value_combination,
				constant_parameters=self.constant_projectile_parameters,
				variable_names=list(
					self.variable_projectile_parameters.keys()))
			simulation_kwargs = get_kwargs(
				variable_value_combination=variable_simulation_parameter_value_combination,
				constant_parameters=self.constant_simulation_parameters,
				variable_names=list(
					self.variable_simulation_parameters.keys()))
			projectile = ProjectileSphereConfiguration()
			projectile.initialize(
				**projectile_kwargs)
			sim = ProjectileMotionSimulationConfiguration()
			sim.initialize_visual_settings(
				tick_size=self.visual_settings.tick_size,
				label_size=self.visual_settings.label_size,
				text_size=self.visual_settings.text_size,
				cell_size=self.visual_settings.cell_size,
				title_size=self.visual_settings.title_size)
			sim.update_save_directory(
				path_to_save_directory=self.visual_settings.path_to_save_directory)
			sim.initialize_simulation(
				projectile=projectile,
				**simulation_kwargs)
			sim.run_simulation(
				**ivp_kwargs)
			return sim			

		variable_projectile_parameter_value_combinations = get_variable_projectile_parameter_value_combinations()
		variable_simulation_parameter_value_combinations = get_variable_simulation_parameter_value_combinations()
		simulations = list()
		if (self.number_projectile_variables > 0) and (self.number_simulation_variables > 0):
			for variable_projectile_parameter_value_combination in variable_projectile_parameter_value_combinations:
				for variable_simulation_parameter_value_combination in variable_simulation_parameter_value_combinations:
					sim = get_simulation(
						variable_projectile_parameter_value_combination=variable_projectile_parameter_value_combination,
						variable_simulation_parameter_value_combination=variable_simulation_parameter_value_combination,
						**ivp_kwargs)
					simulations.append(
						sim)
		elif (self.number_projectile_variables > 0) and (self.number_simulation_variables == 0):
			for variable_projectile_parameter_value_combination in variable_projectile_parameter_value_combinations:
				sim = get_simulation(
					variable_projectile_parameter_value_combination=variable_projectile_parameter_value_combination,
					**ivp_kwargs)
				simulations.append(
					sim)
		elif (self.number_projectile_variables == 0) and (self.number_simulation_variables > 0):
			for variable_simulation_parameter_value_combination in variable_simulation_parameter_value_combinations:
				sim = get_simulation(
					variable_simulation_parameter_value_combination=variable_simulation_parameter_value_combination,
					**ivp_kwargs)
				simulations.append(
					sim)
		else:
			raise ValueError("invalid self.number_variables: {}".format(self.number_variables))
		number_simulations = len(
			simulations)
		self._simulations = simulations
		self._number_simulations = number_simulations

class BaseProjectileMotionEnsembleConfiguration(BaseProjectileMotionEnsembleParameterConfiguration):

	def __init__(self):
		super().__init__()

	def initialize_ensemble(self, radius, mass, drag_coefficient, number_time_steps, dt, launch_position, launch_angle, launch_speed, air_density, g_acceleration, mass_surface, radius_surface, **ivp_kwargs):
		self.initialize_projectile_parameters(
			radius=radius,
			mass=mass,
			drag_coefficient=drag_coefficient)
		self.initialize_simulation_parameters(
			number_time_steps=number_time_steps,
			dt=dt,
			launch_position=launch_position,
			launch_angle=launch_angle,
			launch_speed=launch_speed,
			air_density=air_density,
			g_acceleration=g_acceleration,
			mass_surface=mass_surface,
			radius_surface=radius_surface)
		self.initialize_variable_numbers()
		self.initialize_simulations(
			**ivp_kwargs)

##