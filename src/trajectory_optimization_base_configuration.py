import numpy as np
from scipy.optimize import minimize_scalar, minimize
from projectile_configuration import ProjectileSphereConfiguration
from simulation_configuration import ProjectileMotionSimulationConfiguration
from ensemble_base_configuration import BaseProjectileMotionEnsembleConfiguration
from ensemble_configuration import ProjectileMotionEnsembleConfiguration


class BaseTrajectoryOptimizationParameterization(BaseProjectileMotionEnsembleConfiguration):

	def __init__(self):
		super().__init__()
		self._variable_projectile_parameter_bounds = None
		self._variable_simulation_parameter_bounds = None
		self._variable_bounds = None

	@property
	def variable_projectile_parameter_bounds(self):
		return self._variable_projectile_parameter_bounds
	
	@property
	def variable_simulation_parameter_bounds(self):
		return self._variable_simulation_parameter_bounds
	
	@property
	def variable_bounds(self):
		return self._variable_bounds

	@staticmethod
	def get_base_constant_and_variable_parameters(*args, **kwargs):
		raise ValueError("this method exists for the parent class only")

	@staticmethod
	def initialize_projectile_parameters(*args, **kwargs):
		raise ValueError("this method exists for the parent class only")

	@staticmethod
	def initialize_simulation_parameters(*args, **kwargs):
		raise ValueError("this method exists for the parent class only")

	@staticmethod
	def initialize_simulations(*args, **kwargs):
		raise ValueError("this method exists for the parent class only")

	@staticmethod
	def initialize_variable_numbers(*args, **kwargs):
		raise ValueError("this method exists for the parent class only")

	def initialize_constants_and_optimization_variables(self, launch_angle, launch_speed, radius, mass, drag_coefficient, launch_position, air_density, g_acceleration, mass_surface, radius_surface, number_time_steps, dt):
		
		def get_variable_parameters_and_bounds(parameter_names_and_values):
			constant_parameters = dict()
			variable_parameters = list()
			variable_bounds = list()
			for parameter_name_and_value in parameter_names_and_values:
				(name, value) = parameter_name_and_value
				if value is not None:
					if isinstance(value, (int, float)):
						key = self.convert_parameter_to_key(
							parameter=name)
						constant_parameters[key] = value
					elif isinstance(value, (tuple, list, np.ndarray)):
						number_values = len(
							value)
						if number_values != len(["lower bound", "upper bound"]):
							raise ValueError("invalid len(value) at name={}: {}".format(name, number_values))
						if not isinstance(value[0], (int, float)):
							raise ValueError("invalid type(value[0]) at name={}: {}".format(name, type(value[0])))
						if not isinstance(value[1], (int, float)):
							raise ValueError("invalid type(value[1]) at name={}: {}".format(name, type(value[1])))
						if np.isinf(value[0]) or np.isnan(value[0]):
							raise ValueError("invalid value[0] at name={}: {}".format(name, value[0]))
						if np.isinf(value[1]) or np.isnan(value[1]):
							raise ValueError("invalid value[0] at name={}: {}".format(name, value[1]))
						(lower_bound, upper_bound) = value
						if lower_bound >= upper_bound:
							raise ValueError("invalid value: {}".format(value))
						variable_parameters.append(
							name)
						variable_bounds.append(
							(lower_bound, upper_bound))
			return constant_parameters, variable_parameters, variable_bounds

		def get_other_constant_simulation_parameters(number_time_steps, dt, launch_position, g_acceleration, mass_surface, radius_surface):
			other_constant_simulation_parameters = dict()
			other_constant_simulation_parameters["number time-steps"] = number_time_steps
			other_constant_simulation_parameters["dt"] = dt
			other_constant_simulation_parameters["launch position"] = launch_position
			other_constant_simulation_parameters["g-acceleration"] = g_acceleration
			other_constant_simulation_parameters["mass surface"] = mass_surface
			other_constant_simulation_parameters["radius surface"] = radius_surface
			return other_constant_simulation_parameters

		projectile_parameter_names_and_values = (
			("radius", radius),
			("mass", mass),
			("drag coefficient", drag_coefficient))
		simulation_parameter_names_and_values = (
			("launch angle", launch_angle),
			("launch speed", launch_speed),
			("air density", air_density))
		constant_projectile_parameters, variable_projectile_parameters, variable_projectile_parameter_bounds = get_variable_parameters_and_bounds(
			parameter_names_and_values=projectile_parameter_names_and_values)
		constant_simulation_parameters, variable_simulation_parameters, variable_simulation_parameter_bounds = get_variable_parameters_and_bounds(
			parameter_names_and_values=simulation_parameter_names_and_values)
		other_constant_simulation_parameters = get_other_constant_simulation_parameters(
			number_time_steps=number_time_steps,
			dt=dt,
			launch_position=launch_position,
			g_acceleration=g_acceleration,
			mass_surface=mass_surface,
			radius_surface=radius_surface)
		constant_simulation_parameters.update(
			other_constant_simulation_parameters)
		variable_bounds = variable_projectile_parameter_bounds + variable_simulation_parameter_bounds
		number_projectile_variables = len(
			variable_projectile_parameters)
		number_simulation_variables = len(
			variable_simulation_parameters)
		number_variables = number_projectile_variables + number_simulation_variables
		if number_variables == 0:
			raise ValueError("at least one variable parameter is required for the ensemble")
		self._constant_projectile_parameters = constant_projectile_parameters
		self._constant_simulation_parameters = constant_simulation_parameters
		self._variable_simulation_parameters = variable_simulation_parameters
		self._variable_projectile_parameters = variable_projectile_parameters
		self._number_projectile_variables = number_projectile_variables
		self._number_simulation_variables = number_simulation_variables
		self._number_variables = number_variables
		self._variable_projectile_parameter_bounds = variable_projectile_parameter_bounds
		self._variable_simulation_parameter_bounds = variable_simulation_parameter_bounds
		self._variable_bounds = variable_bounds

class BaseTrajectoryOptimizationMethodsConfiguration(BaseTrajectoryOptimizationParameterization):

	def __init__(self):
		super().__init__()
		self._objective_quantity = None
		self._get_objective = None
		self._optimal_objective_value = None
		self._optimal_parameters_value = None
		self._optimal_simulation = None
		self._optimization_information = None

	@property
	def objective_quantity(self):
		return self._objective_quantity
		
	@property
	def get_objective(self):
		return self._get_objective
	
	@property
	def optimal_objective_value(self):
		return self._optimal_objective_value

	@property
	def optimal_parameters_value(self):
		return self._optimal_parameters_value

	@property
	def optimal_simulation(self):
		return self._optimal_simulation
	
	@property
	def optimization_information(self):
		return self._optimization_information

	@staticmethod
	def select_optimization_objective(objective_quantity):

		def get_t_at_peak(sim):
			value = float(
				sim.parameterization_at_peak["t"])
			objective_value = -1 * value
			return objective_value

		def get_z_at_peak(sim):
			value = float(
				sim.parameterization_at_peak["z"])
			objective_value = -1 * value
			return objective_value

		def get_adjusted_height_at_peak(sim):
			value = float(
				sim.adjusted_height_at_peak)
			objective_value = -1 * value
			return objective_value

		def get_t_at_ground(sim):
			value = float(
				sim.parameterization_at_ground["t"])
			objective_value = -1 * value
			return objective_value

		def get_x_at_ground(sim):
			value = float(
				sim.parameterization_at_ground["x"])
			objective_value = -1 * value
			return objective_value

		def get_range_of_trajectory(sim):
			value = float(
				sim.range_of_trajectory)
			objective_value = -1 * value
			return objective_value

		def get_arc_length_of_trajectory(sim):
			value = float(
				sim.arc_length_of_trajectory)
			objective_value = -1 * value
			return objective_value

		def get_speed(sim):
			value = np.nanmax(
				sim.v)
			objective_value = -1 * value
			return objective_value

		def get_analytic_terminal_speed(sim):
			value = np.nanmax(
				sim.analytic_terminal_speed)
			objective_value = -1 * value
			return objective_value

		mapping = {
			"longest t at peak" : get_t_at_peak,
			"longest z at peak" : get_z_at_peak,
			"longest adjusted height at peak" : get_adjusted_height_at_peak,
			"longest t at ground" : get_t_at_ground,
			"longest x at ground" : get_x_at_ground,
			"longest range of trajectory" : get_range_of_trajectory,
			"longest arc-length of trajectory" : get_arc_length_of_trajectory,
			"fastest v" : get_speed,
			"fastest analytic terminal speed" : get_analytic_terminal_speed}
		if objective_quantity not in mapping.keys():
			raise ValueError("invalid objective_quantity: {}".format(objective_quantity))
		get_value_at_objective = mapping[objective_quantity]
		return get_value_at_objective

	def initialize_optimization_objective(self, objective_quantity, **ivp_kwargs):
		
		def get_projectile_kwargs(variables):
			kwargs = dict()
			for parameter, constant_value in self.constant_projectile_parameters.items():
				key = self.convert_parameter_to_key(
					parameter=parameter)
				kwargs[key] = constant_value
			if self.number_projectile_variables > 0:
				indices = list(
					range(
						self.number_projectile_variables))
				for parameter, index_at_variable in zip(self.variable_projectile_parameters, indices):
					key = self.convert_parameter_to_key(
						parameter=parameter)
					if self.number_variables == 1:
						kwargs[key] = variables
					else:
						kwargs[key] = variables[index_at_variable]
			return kwargs

		def get_simulation_kwargs(variables):
			kwargs = dict()
			for parameter, constant_value in self.constant_simulation_parameters.items():
				key = self.convert_parameter_to_key(
					parameter)
				kwargs[key] = constant_value
			if self.number_simulation_variables > 0:
				indices = list(
					range(
						self.number_projectile_variables,
						self.number_projectile_variables + self.number_simulation_variables + 1))
				for parameter, index_at_variable in zip(self.variable_simulation_parameters, indices):
					key = self.convert_parameter_to_key(
						parameter)
					if self.number_variables == 1:
						kwargs[key] = variables
					else:
						kwargs[key] = variables[index_at_variable]
			return kwargs

		def get_objective_value(get_value_at_objective, variables, **ivp_kwargs):
			projectile_kwargs = get_projectile_kwargs(
				variables=variables)
			projectile = ProjectileSphereConfiguration()
			projectile.initialize(
				**projectile_kwargs)
			simulation_kwargs = get_simulation_kwargs(
				variables=variables)
			sim = ProjectileMotionSimulationConfiguration()
			sim.initialize_simulation(
				projectile=projectile,
				**simulation_kwargs)
			sim.run_simulation(
				**ivp_kwargs)
			objective_value = get_value_at_objective(
				sim=sim)
			return objective_value			

		get_value_at_objective = self.select_optimization_objective(
			objective_quantity=objective_quantity)
		get_objective = lambda variables : get_objective_value(
			get_value_at_objective=get_value_at_objective,
			variables=variables,
			**ivp_kwargs)
		self._objective_quantity = objective_quantity
		self._get_objective = get_objective		

	def initialize_optimization_results(self, **opt_kwargs):
		if self.number_variables is None:
			raise ValueError("cannot optimize trajectory with zero variables")
		if self.number_variables == 0:
			raise ValueError("cannot optimize trajectory with zero variables")
		if self.number_variables == 1:
			bounds = tuple(
				self.variable_bounds[0])
			bracket = bounds
			optimization_result = minimize_scalar(
				self.get_objective,
				bracket=bracket,
				bounds=bounds,
				**opt_kwargs)
		else: # elif self.number_variables > 1:
			initial_parameter_values = [
				np.mean(parameter_bounds)
					for parameter_bounds in self.variable_bounds]
			bounds = tuple(
				self.variable_bounds)
			optimization_result = minimize(
				self.get_objective,
				initial_parameter_values,
				bounds=bounds,
				**opt_kwargs)
		if not optimization_result.success:
			raise ValueError(optimization_result)
		objective_value = -1 * optimization_result.fun
		optimization_information = {
			"result" : optimization_result,
			"parameters" : optimization_result.x,
			"objective value" : objective_value}
		self._optimal_objective_value = objective_value
		self._optimal_parameters_value = optimization_result.x
		self._optimization_information = optimization_information

	def initialize_optimal_trajectory(self, **ivp_kwargs):
		
		def get_projectile_kwargs():
			kwargs = dict()
			for parameter, constant_value in self.constant_projectile_parameters.items():
				key = self.convert_parameter_to_key(
					parameter=parameter)
				kwargs[key] = constant_value
			if self.number_projectile_variables > 0:
				optimal_parameters = self.optimization_information["parameters"]
				indices = list(
					range(
						self.number_projectile_variables))
				for parameter, index_at_variable in zip(self.variable_projectile_parameters, indices):
					key = self.convert_parameter_to_key(
						parameter=parameter)
					if self.number_variables == 1:
						kwargs[key] = optimal_parameters
					else:
						kwargs[key] = optimal_parameters[index_at_variable]
			return kwargs

		def get_simulation_kwargs():
			kwargs = dict()
			for parameter, constant_value in self.constant_simulation_parameters.items():
				key = self.convert_parameter_to_key(
					parameter)
				kwargs[key] = constant_value
			if self.number_simulation_variables > 0:
				optimal_parameters = self.optimization_information["parameters"]
				indices = list(
					range(
						self.number_simulation_variables))
				for parameter, index_at_variable in zip(self.variable_simulation_parameters, indices):
					key = self.convert_parameter_to_key(
						parameter=parameter)
					if self.number_variables == 1:
						kwargs[key] = optimal_parameters
					else:
						kwargs[key] = optimal_parameters[index_at_variable]
			return kwargs

		projectile_kwargs = get_projectile_kwargs()
		projectile = ProjectileSphereConfiguration()
		projectile.initialize(
			**projectile_kwargs)
		simulation_kwargs = get_simulation_kwargs()
		optimal_simulation = ProjectileMotionSimulationConfiguration()
		optimal_simulation.initialize_visual_settings()
		optimal_simulation.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		optimal_simulation.initialize_simulation(
			projectile=projectile,
			**simulation_kwargs)
		optimal_simulation.run_simulation(
			**ivp_kwargs)
		self._optimal_simulation = optimal_simulation

class BaseTrajectoryOptimizationConfiguration(BaseTrajectoryOptimizationMethodsConfiguration):

	def __init__(self):
		super().__init__()
		self._ensemble = None

	@property
	def ensemble(self):
		return self._ensemble

	def initialize_optimization(self, objective_quantity, **ivp_kwargs):
		self.initialize_optimization_objective(
			objective_quantity=objective_quantity,
			**ivp_kwargs)
		if self.number_variables == 1:
			self.initialize_optimization_results(
				method="Bounded",
				# method="Brent",
				# method="Golden",
				)
		else:
			self.initialize_optimization_results(
				method="Nelder-Mead",
				# method="COBYLA",
				# method="L-BFGS-B",
				)
		self.initialize_optimal_trajectory(
			**ivp_kwargs)

	def initialize_ensemble(self, number_samples_per_variable, **ivp_kwargs):
		if not isinstance(number_samples_per_variable, int):
			raise ValueError("invalid type(number_samples_per_variable): {}".format(type(number_samples_per_variable)))
		if number_samples_per_variable <= 1:
			raise ValueError("invalid number_samples_per_variable: {}".format(number_samples_per_variable))
		kwargs = dict()
		for name, constant_value in self.constant_projectile_parameters.items():
			key = self.convert_parameter_to_key(
				parameter=name)
			kwargs[key] = constant_value
		for name, constant_value in self.constant_simulation_parameters.items():
			key = self.convert_parameter_to_key(
				parameter=name)
			kwargs[key] = constant_value
		for name, bounds in zip(self.variable_simulation_parameters, self.variable_simulation_parameter_bounds):
			key = self.convert_parameter_to_key(
				parameter=name)
			(lower_bound, upper_bound) = bounds
			sample_values = np.linspace(
				lower_bound,
				upper_bound,
				number_samples_per_variable)
			kwargs[key] = sample_values
		for name, bounds in zip(self.variable_projectile_parameters, self.variable_projectile_parameter_bounds):
			key = self.convert_parameter_to_key(
				parameter=name)
			(lower_bound, upper_bound) = bounds
			sample_values = np.linspace(
				lower_bound,
				upper_bound,
				number_samples_per_variable)
			kwargs[key] = sample_values
		kwargs.update(
			**ivp_kwargs)
		ensemble = ProjectileMotionEnsembleConfiguration()
		ensemble.initialize_visual_settings()
		ensemble.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		ensemble.initialize_ensemble(
			**kwargs)
		self._ensemble = ensemble

##