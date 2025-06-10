import numpy as np


class BaseProjectileSphereConfiguration():

	def __init__(self):
		super().__init__()
		self._name = None
		self._radius = None
		self._cross_sectional_area = None
		self._volume = None
		self._mass = None
		self._density = None
		self._drag_coefficient = None

	@property
	def name(self):
		return self._name
	
	@property
	def radius(self):
		return self._radius
	
	@property
	def cross_sectional_area(self):
		return self._cross_sectional_area
	
	@property
	def volume(self):
		return self._volume

	@property
	def mass(self):
		return self._mass

	@property
	def density(self):
		return self._density

	@property
	def drag_coefficient(self):
		return self._drag_coefficient

	def initialize_name(self):
		name = "Projectile Sphere"
		self._name = name

	def initialize_radius(self, radius):
		if not isinstance(radius, (int, float)):
			raise ValueError("invalid type(radius): {}".format(type(radius)))
		if radius <= 0:
			raise ValueError("radius should be greater than zero")
		self._radius = radius

	def initialize_cross_sectional_area(self):
		cross_sectional_area = np.pi * self.radius * self.radius
		self._cross_sectional_area = cross_sectional_area

	def initialize_volume(self):
		volume = (4 / 3) * np.pi * self.radius ** 3
		self._volume = volume

	def initialize_mass(self, mass):
		if not isinstance(mass, (int, float)):
			raise ValueError("invalid type(mass): {}".format(type(mass)))
		if mass <= 0:
			raise ValueError("invalid mass: {}".format(mass))
		self._mass = mass

	def initialize_density(self):
		density = self.mass / self.volume
		self._density = density

	def initialize_drag_coefficient(self, drag_coefficient):
		if not isinstance(drag_coefficient, (int, float)):
			raise ValueError("invalid type(drag_coefficient): {}".format(type(drag_coefficient)))
		if drag_coefficient < 0:
			raise ValueError("invalid drag_coefficient: {}".format(drag_coefficient))
		self._drag_coefficient = drag_coefficient

class ProjectileSphereConfiguration(BaseProjectileSphereConfiguration):

	def __init__(self):
		super().__init__()

	def initialize(self, radius, mass, drag_coefficient):
		self.initialize_name()
		self.initialize_radius(
			radius=radius)
		self.initialize_cross_sectional_area()
		self.initialize_volume()
		self.initialize_mass(
			mass=mass)
		self.initialize_density()
		self.initialize_drag_coefficient(
			drag_coefficient=drag_coefficient)

	def get_projectile_label(self):
		label = r"Projectile Sphere ($r={:,.2f}$ $m$)".format(
			self.radius)
		return label

	def get_value(self, parameter):
		parameters = [
			"radius",
			"mass",
			"volume",
			"density",
			"drag coefficient"]		
		if parameter not in parameters:
			raise ValueError("invalid parameter: {}".format(parameter))
		key = parameter.replace(
			"-",
			"_")
		key = key.replace(
			" ",
			"_")
		unmodified_value = getattr(
			self,
			key)
		if isinstance(unmodified_value, int):
			value = int(
				unmodified_value)
		elif isinstance(unmodified_value, float):
			value = float(
				unmodified_value)
		else:
			raise ValueError("invalid type(self.{}): {}".format(key, type(unmodified_value)))
		return value

##