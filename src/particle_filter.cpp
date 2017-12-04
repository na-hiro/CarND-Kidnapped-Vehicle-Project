/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set number of particles
	num_particles = 100;

	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);
	// Create normal distributions for y and theta
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		p.id = i;

		particles.push_back(p);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (int i = 0; i < num_particles; i++) {

		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		double new_x;
		double new_y;
		double new_theta;

		// calculate new state
		if (fabs(yaw_rate) < 0.0001) {
			new_x = particle_x + velocity * cos(particle_theta) * delta_t;
			new_y = particle_y + velocity * sin(particle_theta) * delta_t;
			new_theta = particle_theta;
		}
		else {
			new_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
			new_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
			new_theta = particle_theta + (yaw_rate * delta_t);
		}

		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i=0; i<observations.size();i++){
			double mindist = numeric_limits<double>::max();

			int id = -1;

			for (auto pred : predicted) {

					double curr_dist = dist(observations[i].x, observations[i].y, pred.x, pred.y);

					if (curr_dist < mindist) {
							mindist = curr_dist;
							id = pred.id;
					}
			}

			// set the observation's id to the nearest predicted landmark's id
			observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (unsigned int i=0; i < num_particles; i++) {

		// get particle's position and coordinates
  	double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
		//printf("x, y, theta: %lf, %lf, %lf\n", x, y, theta);

		// create a vector of predicted landmark pos.
    vector<LandmarkObs> predictions;

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

    	// get id and x,y coordinates of landmark
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      // consider landmarks within only sensor range
      if (fabs(lm_x - x) <= sensor_range && fabs(lm_y - y) <= sensor_range) {
      	// add prediction to vector
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
		}
		//printf("obs_size: %d\n", (int)observations.size());

		vector<LandmarkObs> obesrvations_tf;
    for (unsigned int k=0; k < observations.size(); k++) {
			// transform to map x,y coordinate
      	double x_map = x + (cos(theta) * observations[k].x) - (sin(theta) * observations[k].y);
        double y_map = y + (sin(theta) * observations[k].x) + (cos(theta) * observations[k].y);
        obesrvations_tf.push_back(LandmarkObs{-1, x_map, y_map});
		}

		// association
    dataAssociation(predictions, obesrvations_tf);

		// Initialize weights
    weights[i] = 1;

		// calculate normalization term
    double gauss_norm = 1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);

    for (unsigned int k=0; k < obesrvations_tf.size(); k++) {

			// get observation
			double x_obs = obesrvations_tf[k].x;
			double y_obs = obesrvations_tf[k].y;

    	for (unsigned int j=0; j < predictions.size(); j++) {
    		if (predictions[j].id == obesrvations_tf[k].id) {

					// get associated prediction coordinates
        	double mu_x = predictions[j].x;
        	double mu_y = predictions[j].y;

					// calculate exponent
        	double exponent = pow(x_obs - mu_x, 2)/(2 * pow(std_landmark[0],2)) + pow(y_obs - mu_y,2)/(2 * pow(std_landmark[1], 2));
					// calculate weight using normalization terms and exponent
        	weights[i] = weights[i] * gauss_norm * exp(-exponent);
      	}
			}
	  }
		// Particle's Final Weight
  	particles[i].weight = weights[i];
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> distribution(weights.begin(), weights.end());

	std::vector<Particle> new_particles;
	for (unsigned int i=0; i<num_particles; i++) {
			int rd_id = distribution(gen);
			new_particles.push_back(std::move(particles[rd_id]));
	}

	particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

		//Clear the previous associations
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
