# Repo:    projectile-motion

The purpose of this code is to model, analyze, and optimize the trajectory of an airborne projectile that is subject to gravitational and drag forces.

## Description

"What goes up must come down" is a famous quote that is sometimes misattributed to Isaac Newton. When applied to physics, this phrase is often interpreted as meaning that objects of mass that are airborne appear to fall down to the surface of the Earth due to the force of gravity. 

I will use bold-face font to denote vector quantities. Suppose one launches a projectile sphere of mass $m$ and radius $r$ - moving at velocity $𝒗$ in the $xz$-plane - and that the force of gravity $𝐅_{𝒈}$ acts upon the projectile.

$𝐅_{𝒈} = - mg\hat{𝒛}$

where $a_{g} = g \approx 9.8 \frac{m}{s^2}$

We can use a system of differential equations to describe the motion of the projectile.

$m\frac{d^2x}{dt^2} = 0$

$m\frac{d^2z}{dt^2} = - mg$

The solution to this system of differential equations is analytic.

$x(t) = x_{0}$ $+$ $v_{x}(t)$ $\cdot$ $t$

$z(t) = z_{0}$ $+$ $v_{z}(t)$ $\cdot$ $t$ $-$ $\frac{1}{2}$ $g$ $t^{2}$

where

$v_{x}(t) = \frac{dx}{dt}|_{t}$

$v_{z}(t) = \frac{dz}{dt}|_{t}$

$𝒗(t) = v_{x}(t)\hat{𝒙} + v_{z}(t)\hat{𝒛}$

<img src="output/example_06-ensemble_with_variable_launch_angle_at_g_of_z/ProjectileMotionEnsemble-Var_launch_angle.png" title="" alt="example-ensemble_of_variable_launch_angle" data-align="center">

Given the initial launch speed $v_{0}$ $=$ $\sqrt{𝒗 \cdot 𝒗}$ and initial launch angle $\phi_{0}$ (both evaluated at $t=0$), the initial condition is obtained for this system of differential equations.

$v_{x, t=0}$ $\equiv$ $\frac{dx}{dt}$ (evaluated at $t=0$) = $v_{0} \cos{\phi_{0}}$

$v_{z, t=0}$ $\equiv$ $\frac{dz}{dt}$ (evaluated at $t=0$) = $v_{0} \sin{\phi_{0}}$

One can verify that this model obeys the principle of conservation of energy. Note that the moment that the projectile is at its peak is the same moment that the kinetic energy is at its minimum.

<img src="output/example_01-simulation_without_drag_at_constant_g/ProjectileMotionSimulation-Energies_VS_Time-wPeakTime.png" title="" alt="example-energy_conservation" data-align="center">

<img title="" src="output/example_01-simulation_without_drag_at_constant_g/ProjectileMotionSimulation-EnergyVar_VS_Time-wPeakTime.png" alt="example-negligible_energy_variance" data-align="center">

The accuracy of the model improves if one accounts for air resistance, which produces a drag force that acts in a manner similar to friction in the sense that energy is not conserved. There are a variety of models to account for the drag force; this code uses the quadratic drag model. To further improve accuracy, one could use piece-wise velocity-dependent drag models. According to the quadratic drag model, the drag force is given by

$𝐅_{𝑫} = - \frac{1}{2} C_{d} \rho_{air} A_{c} v^{2} \hat{𝒗}$

where

$C_{d}$ is the drag coefficient of the projectile

$\rho_{air}$ is the air density

$A_{c}$ is the cross-sectional area of the projectile (sphere $\implies$ $A_{c} = \pi r^2$)

<img src="output/example_03-simulation_with_drag_at_constant_g/ProjectileMotionSimulation-SpeedsAndVelocities_VS_Time-wPeakTime.png" title="" alt="example-air_drag_and_terminal_velocity" data-align="center">

The model can be further improved by accounting for the altitude $z$ of the projectile when calculating its acceleration due to gravity (as opposed to taking the value $g \approx 9.8 \frac{m}{s^2}$ to be constant). 

$g \equiv g(z) = \frac{GM}{(R + z)^2}$ 

$g(z=0) = 9.8 \frac{m}{s^2}$

where

$G \approx 6.676 \times 10^{-11} \frac{m^3}{kg \cdot s^2}$ is Newton's gravitational constant

$M \approx 5.972 \times 10^{24}$ kg is the mass of Earth

$R \approx 6371$ km is the radius of Earth

There is no general analytic solution to account for air resistance and altitude-dependent acceleration due to gravity, but the solution can be obtained numerically; this code uses `scipy.integrate.solve_ivp` to solve this system of differential equations. 

$m \frac{d^{2}x}{dt^2} = - k \sqrt{(\frac{dx}{dt})^{2} + (\frac{dz}{dt})^{2}} \frac{dx}{dt}$

$m \frac{d^{2}z}{dt^2} = - k \sqrt{(\frac{dx}{dt})^{2} + (\frac{dz}{dt})^{2}} \frac{dz}{dt} - mg$

where

$k = \frac{1}{2} C_{d} \rho_{air} A_{c}$

One can use this code to determine the optimal launch angle $\phi_{0}$; here, optimal can refer to:

* the trajectory that maximizes the height of the projectile at its peak

* the trajectory that maximizes the range of the projectile

* the trajectory that maximizes the amount of time that is required for the projectile to reach its peak

* the trajectory that maximizes the amount of time that the projectile stays airborne

* the trajectory of the largest possible arc-length

To visually confirm that the calculated extremum is a true maximum, one can plot the optimized parameters against an ensemble of bounded values of $\phi_{0}$. The plot below shows this for the trajectory of the largest possible arc-length.

![example-optimal_trajectory_with_variable_g](output/example_12-optimization_with_variable_launch_angle_at_g_of_z/TrajectoryOptimization-LongestArcLengthOfTrajectory_VS_1_Var-LaunchAngle.png)

## Getting Started

### Dependencies

* Python 3.9.6
* numpy == 1.26.4
* matplotlib == 3.9.4
* scipy == 1.13.1
* itertools (default)
* re (default)

### Executing program

* Download this repository to your local computer

* `cd` into the `src` directory, select any of the example files, modify `path_to_save_directory`, and then run the script.

## Version History

* 0.1
  * Initial Release

## To-Do
* fix formatting of legend in plots

* compare with projectiles of other shapes (account for rotation if $A_{c}$ is not constant)

## License

This project is licensed under the Apache License - see the LICENSE file for details.
