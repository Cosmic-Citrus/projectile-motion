# Repo:    projectile-motion

The purpose of this code is to analyze and optimize the trajectory of a spherical projectile.

## Description

"What goes up must come down" is a famous quote that is often misattributed to Isaac Newton. When applied to physics, this phrase is often interpreted as meaning that objects of mass appear to fall back down because of gravity. One can account for the force of gravity to model the trajectory of a projectile.

Suppose one launches a projectile sphere of mass $m$ and radius $r$ - moving at velocity $v$ in the $xz$-plane - and that the force of gravity $F_{g}$ acts upon the projectile. 

$\bold{F_{g}} = - mg\bold{\hat{z}}$

where $g \approx 9.81 \frac{m}{s^2}$

We can use a system of differential equations to describe the motion of the projectile.

$m\frac{d^2x}{dt^2} = 0$

$m\frac{d^2z}{dt^2} = - mg$

The solution to this system of differential equations is analytic.

$x(t) = x_{0}$ $+$ $v_{x}(t)$ $\cdot$ $t$

$z(t) = z_{0}$ $+$ $v_{z}(t)$ $\cdot$ $t$ $-$ $\frac{1}{2}$ $g$ $t^{2}$

where

$v_{x}(t) = \frac{dx}{dt}|_{t}$

$v_{z}(t) = \frac{dz}{dt}|_{t}$

<img src="output/example_06-ensemble_with_variable_launch_angle_at_g_of_z/ProjectileMotionEnsemble-Var_launch_angle.png" title="" alt="example-ensemble_of_variable_launch_angle" data-align="center">

Given the initial launch speed $v_{0}$ $=$ $\sqrt{\bold{v} \cdot \bold{v}}$ (evaluated at $t=0$) and initial launch angle $\phi_{0}$, the initial condition is obtained for this system of differential equations.

$v_{x, t=0}$ $\equiv$ $\frac{dx}{dt}$ (evaluated at $t=0$) = $v_{0} \cos{\phi_{0}}$

$v_{z, t=0}$ $\equiv$ $\frac{dz}{dt}$ (evaluated at $t=0$) = $v_{0} \sin{\phi_{0}}$

One can verify that this model obeys the principle of conservation of energy.

<img src="output/example_01-simulation_without_drag_at_constant_g/ProjectileMotionSimulation-Energies_VS_Time-wPeakTime.png" title="" alt="example-energy_conservation" data-align="center">

<img title="" src="output/example_01-simulation_without_drag_at_constant_g/ProjectileMotionSimulation-EnergyVar_VS_Time-wPeakTime.png" alt="example-negligible_energy_variance" data-align="center">

The accuracy of the model improves if one accounts for air resistance, which produces a drag force that acts in a manner similar to friction - this means that energy is not conserved. There are a variety of models to account for the drag force; this example uses the quadratic drag model. To further improve accuracy, one could use piece-wise velocity-dependent drag models. According to the quadratic drag model, the drag force is given by

$\bold{F_{D}} = - \frac{1}{2} C \rho A_{c} v^2 \bold{\hat{v}}$

where

$C$ is the drag coefficient of the projectile

$\rho$ is the air density

$A_{c}$ is the cross-sectional area of the projectile (sphere $\implies$ $A_{c} = \pi r^2$)

<img src="output/example_03-simulation_with_drag_at_constant_g/ProjectileMotionSimulation-SpeedsAndVelocities_VS_Time-wPeakTime.png" title="" alt="example-air_drag_and_terminal_velocity" data-align="center">

The model can be further improved further by accounting for the altitude $z$ of the projectile when calculating its acceleration due to gravity (as opposed to taking the value $g \approx 9.81 \frac{m}{s^2}$ to be constant). 

$g \equiv g(z) = \frac{GM}{(R + z)^2}$

where

$G \approx 6.676 \times 10^{-11} \frac{m^3}{kg \cdot s^2}$ is Newton's gravitational constant

$M \approx 5.972 \times 10^{24}$ kg is the mass of Earth

$R \approx 6371$ km is the radius of Earth

There is no general analytic solution to account for air resistance and altitude-dependent acceleration due to gravity, but the solution can be obtained numerically; this code uses `scipy.integrate.solve_ivp` to solve this system of differential equations. 

$m \frac{d^{2}x}{dt^2} = - k \sqrt{(\frac{dx}{dt})^{2} + (\frac{dz}{dt})^{2}} \frac{dx}{dt}$

$m \frac{d^{2}z}{dt^2} = - k \sqrt{(\frac{dx}{dt})^{2} + (\frac{dz}{dt})^{2}} \frac{dz}{dt} - mg$

where

$k = \frac{1}{2} C \rho A_{c}$

One can use this code to determine the optimal launch angle $\phi_{0}$; here, optimal can refer to:

* the trajectory that maximizes the height of the projectile at its peak

* the trajectory that maximizes the range of the projectile

* the trajectory that maximizes the amount of time that is required for the projectile to reach its peak

* the trajectory that maximizes the amount of time that the projectile stays airborne

* the trajectory of the largest possible arc-length (shown below)

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

* compare with projectiles of other shapes

* account for rotation when calculating cross-sectional area of non-spherical projectiles

## License

This project is licensed under the Apache License - see the LICENSE file for details.
