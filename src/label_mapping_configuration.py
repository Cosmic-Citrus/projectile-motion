import re


class BaseLabelMappingConfiguration():

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_label_segments(parameter, value=None):
		combinations = (
			("radius", r"$r$", r"$m$"),
			("side", r"$s$", r"$m$"),
			("length", r"$l$", r"$m$"),
			("width", r"$w$", r"$m$"),
			("height", r"$h$", r"$m$"),
			("drag coefficient", r"$C_{d_{projectile}}$", None),
			("mass", r"$m_{projectile}$", r"$kg$"),
			("density", r"$\rho_{projectile}$", r"$\frac{kg}{m^3}$"),
			("volume", r"$V$", r"$m^3$"),
			("moment of inertia", r"$I_{projectile}$", r"$kg * m^2$"),
			("cross-sectional area", r"$A_{cross-section}$", r"$m^2$"),
			("air density", r"$\rho_{air}$", r"$\frac{kg}{m^3}$"),
			("g-acceleration", r"$a_{gravity}$", r"$\frac{m}{s^2}$"),
			("launch position", r"$(x_{0}, z_{0})$", r"$(m, m)$"),
			("launch angle", r"$\phi_{0}$", r"$rad$"),
			("launch speed", r"$v(t=0)$", r"$\frac{m}{s}$"),
			("number time-steps", r"$N_{t}$", r"time-steps"),
			("time step-size", r"$\Delta t$", r"$s$"),
			("time", r"$t$", r"$s$"),
			("x", r"$x$", r"$m$"),
			("y", r"$y$", r"$m$"),
			("z", r"$z$", r"$m$"),
			("dx/dt", r"$\frac{dx}{dt}$", r"$\frac{m}{s}$"),
			("dy/dt", r"$\frac{dy}{dt}$", r"$\frac{m}{s}$"),
			("dz/dt", r"$\frac{dz}{dt}$", r"$\frac{m}{s}$"),
			("v", r"$v$", r"$\frac{m}{s}$"),
			("analytic terminal speed", r"$v_{terminal}$", r"$\frac{m}{s}$"),
			("potential energy", r"$E_{potential}$", r"$J$"),
			("kinetic energy", r"$E_{kinetic}$", r"$J$"),
			("lagrangian", r"$ℒ$", r"$J$"),
			("hamiltonian", r"$ℋ$", r"$J$"),
			("hamiltonian variance", r"$|\frac{ℋ(t) - ℋ(t=0)}{ℋ(t=0)}|$ $\times$ $100\%$", r"$\%$"),
			("t at peak", r"$t_{peak}$", r"$s$"),
			("z at peak", r"$z_{peak}$", r"$m$"),
			("adjusted height at peak", r"$z_{max} - z_{t=0}$", r"$m$"),
			("t at ground", r"$t_{ground}$", r"$s$"),
			("x at ground", r"$x_{ground}$", r"$m$"),
			("range of trajectory", r"$x_{max} - x_{t=0}$", r"$m$"),
			("arc-length of trajectory", r"$length_{arc}$", r"$m$"),
			)
		parameters = [
			combination[0]
				for combination in combinations]
		if parameter in parameters:
			index_at_parameter = parameters.index(
				parameter)
			label_segments = combinations[index_at_parameter]
			(_, parameter_label, unit_label) = label_segments
		else:
			print(parameter)
			raise ValueError("FF")
		if value is None:
			value_label = r"None"
			if unit_label is None:
				full_label = r"{}".format(
					parameter_label)
			else:
				full_label = r"{} [{}]".format(
					parameter_label,
					unit_label)
		else:
			if isinstance(value, int):
				value_label = r"${:,}$".format(
					value)
			elif isinstance(value, float):
				value_label = r"${:,.2f}$".format(
					value)
			else:
				raise ValueError("invalid type(value): {}".format(type(value)))
			if unit_label is None:
				full_label = r"{} $=$ {}".format(
					parameter_label,
					value_label)
			else:
				full_label = r"{} $=$ {} {}".format(
					parameter_label,
					value_label,
					unit_label)
		label_segments = (
			parameter_label,
			value_label,
			unit_label,
			full_label)
		return label_segments

	@staticmethod
	def get_time_interval_label(t):
		lower_bound = t[0]
		upper_bound = t[-1]
		label = r"${:,.2f}$ $s$ $≤$ $t$ $≤$ ${:,.2f}$ $s$".format(
			lower_bound,
			upper_bound)
		return label

class LabelMappingMethodsConfiguration(BaseLabelMappingConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_condensed_scientific_notation_label(value):

		def get_label_with_base_and_exponent(base, exponent):
			base_with_exponent = f"{base}^{{{exponent}}}"
			return base_with_exponent

		def get_label_with_e(s):
			index_at_e = s.find(
				"e")
			index_at_sign = index_at_e + 1
			factor_in_exponent = 1 if s[index_at_sign] == "+" else -1
			index_at_exponent = index_at_sign + 1
			factor = s[:index_at_e]
			exponent = int(
				s[index_at_exponent:])
			base_with_exponent = get_label_with_base_and_exponent(
				base=10,
				exponent=exponent * factor_in_exponent)
			label = r"${}$ $\times$ ${}$".format(
				factor,
				base_with_exponent)
			return label

		def get_label_without_e(s):
			size = len(
				s)
			index_at_prime_decimal = 1
			is_contains_decimal = (
				"." in s)
			if is_contains_decimal:
				index_at_decimal = s.find(
					".")
				prefix = r"${}.{}$".format(
					s[0],
					s[1:4].replace(
						".",
						""))
			else:
				index_at_decimal = size - 1
				prefix = r"${}.{}$".format(
					s[0],
					s[1:3])
			exponent = index_at_decimal - index_at_prime_decimal
			base_with_exponent = get_label_with_base_and_exponent(
				base=10,
				exponent=exponent)
			label = r"{} $\times$ ${}$".format(
				prefix,
				base_with_exponent)
			# if int(exponent) == float(exponent):
			# 	label = r"{} $\times$ $10^{:,}$".format(
			# 		prefix,
			# 		exponent)
			# else:
			# 	label = r"{} $\times$ $10^{:,.2f}$".format(
			# 		prefix,
			# 		exponent)
			return label

		if not isinstance(value, (int, float)):
			raise ValueError("invalid type(value): {}".format(type(value)))
		s = str(
			value)
		if "e" in s:
			label = get_label_with_e(
				s=s)
		else:
			label = get_label_without_e(
				s=s)
		return label

	@staticmethod
	def delete_between_substrings(s, start_substring, end_substring):
		pattern = re.escape(start_substring) + r".*?" + re.escape(end_substring)
		modified_s = re.sub(
			pattern,
			"",
			s)
		return modified_s

	def get_space_corrected_label(self, top_label, bottom_label=None):
		if bottom_label is None:
			(top_label, bottom_label) = top_label.split(
				"\n")
		modified_top_label = self.delete_between_substrings(
			s=top_label,
			start_substring="$",
			end_substring="$")
		modified_bottom_label = self.delete_between_substrings(
			s=bottom_label,
			start_substring="$",
			end_substring="$")
		number_characters_at_top_label = len(
			modified_top_label)
		number_characters_at_bottom_label = len(
			modified_bottom_label)
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

class LabelMappingConfiguration(LabelMappingMethodsConfiguration):

	def __init__(self):
		super().__init__()

	def get_parameter_label(self, parameter):
		label_segments = self.get_label_segments(
			parameter=parameter,
			value=None)
		parameter_label = label_segments[0]
		return parameter_label

	def get_value_label(self, parameter, value):
		label_segments = self.get_label_segments(
			parameter=parameter,
			value=value)
		value_label = label_segments[1]
		return value_label

	def get_unit_label(self, parameter):
		label_segments = self.get_label_segments(
			parameter=parameter,
			value=None)
		unit_label = label_segments[2]
		return unit_label

	def get_full_label(self, parameter, value=None):
		label_segments = self.get_label_segments(
			parameter=parameter,
			value=value)
		full_label = label_segments[3]
		return full_label

	def get_time_label(self, t, dt, number_time_steps):
		interval_label = self.get_time_interval_label(
			t=t)
		step_size_label = self.get_full_label(
			parameter="time step-size",
			value=dt)
		number_steps_label = self.get_full_label(
			parameter="number time-steps",
			value=number_time_steps)
		label = "{}, {}, {}".format(
			interval_label,
			step_size_label,
			number_steps_label)
		return label

##