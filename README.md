# Particle-Filter
## Introduction and Theoretical Background
Robot localization is a fundamental problem in mobile robotics.  The objectiveis to estimate the pose of a robot, represented by its position and orientation,while  accounting  for  uncertainty  arising  from  sensor  noise  and  imperfect  mo-tion  measurements.   In  this  project,  a  Particle  Filter  (PF)  was  implementedto estimate the pose of a mobile robot navigating in a simulated environmentcontaining multiple AR tags.
There are three steps of particle filter algorithms
* Prediction Step - This stage corre-sponds to computing the prior distribution.
* Update (Correction) Step - This stage incorporates sensor information and produces the posteriordistribution
* Resampling Step - Particles with high weights are duplicated while particles with lowweights are discarded

## Simulation Setup
* Environment Configuration - A  rectangular  room  with  dimensions  of4 m×6 mwas  created  for  the  ex-periments.
* Robot Platform - The mobile robot is a differential-drive platform consisting of two powered frontwheels and a passive caster wheel at the rear.
* Camera Configuration - you can see the camera parameters in the result folder (Particle_Filter_Project_Report.pdf)
  
## Motion Model (Prediction Step)
The motion model applies the robot motion estimated from odometry data to each particle and adds Gaussian noise to the position and orientation changes to model real-world uncertainties. The updated motion is computed relative to each particle's own orientation, and the resulting angle is normalized to the range [−π,π]. Finally, the particles are constrained within the known environment boundaries, preventing invalid pose hypotheses outside the map.

## Sensor Model (Update Step)
The sensor model evaluates how well each particle explains the AR tags observed by the camera. Since all AR tags share the same ID, instead of selecting a single tag, the model considers all tags as potential sources and computes the observation likelihood as the sum of their individual likelihoods. The particle weights are then updated and normalized using the resulting likelihood. Over time, incorrect pose hypotheses are eliminated, allowing the filter to converge to the robot's true position.

## Experimental Results and Discussion
(You can find the result in the results folder) The  experimental  results  demonstrate  the  robustness  and  accuracy  of  theimplemented  algorithm.   Initializing  the  particles  with  a  uniform  distributionallowed  the  system  to  effectively  solve  the  global  localization  problem  with-out requiring a prior pose estimate.  Despite the inherent drift associated withthe odometry-based motion model, the multi-hypothesis sensor model reliablyupdated the particle weights based on range and bearing measurements,  suc-cessfully steering the belief toward the true state.
