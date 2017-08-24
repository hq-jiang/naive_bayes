#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	cout << "Enter training" << endl;
	int num_samples = labels.size();
	int x_dim = data[0].size();
	int y_dim = possible_labels.size();
	
	// Calculate prior
    prior.resize(y_dim);
	for (int i=0; i<num_samples; ++i){
	    for (int j=0; j<y_dim; ++j){
	        if (labels.at(i) == possible_labels.at(j)){
	            prior.at(j) += 1;
	        }
	    }
	}
	cout << "prior" << endl;
	for (int i=0; i<prior.size(); ++i){
	    prior.at(i) /= num_samples;
	    cout << prior.at(i) << "  ";
	}
    cout << endl;
    
	// Calculate likelihood matrix
	mu_likeli.resize(y_dim);
	sigma_likeli.resize(y_dim);
	//mu
	for (int i=0; i<y_dim; ++i){
	    mu_likeli.at(i).resize(x_dim);
	    vector<double> num_subsamples(x_dim);
	    for (int j=0; j<x_dim; ++j){
	        for (int k=0; k<num_samples; ++k){
	            if (labels.at(k)==possible_labels.at(i)){
	                mu_likeli.at(i).at(j) += data.at(k).at(j);
	                num_subsamples.at(j) += 1;
	            } 
	        }
	    }
	    for (int j=0; j<x_dim; ++j){
	        mu_likeli.at(i).at(j) /= num_subsamples.at(j);
	    }
	}
	// sigma
	for (int i=0; i<y_dim; ++i){
	    sigma_likeli.at(i).resize(x_dim);
	    vector<double> num_subsamples(x_dim);
	    for (int j=0; j<x_dim; ++j){
	        for (int k=0; k<num_samples; ++k){
	            if (labels.at(k)==possible_labels.at(i)){
	                sigma_likeli.at(i).at(j) += pow(data.at(k).at(j)-mu_likeli.at(i).at(j), 2);
	                num_subsamples.at(j) += 1;
	            } 
	        }
	    }
	    for (int j=0; j<x_dim; ++j){
	        sigma_likeli.at(i).at(j) /= num_subsamples.at(j);
	    }
	}
	cout << "mu_likelihood" << endl;
	for (int i=0; i<mu_likeli.size(); ++i){
	    for (int j=0; j<mu_likeli.at(i).size(); ++j){
	        cout << mu_likeli.at(i).at(j) << "  ";
	    }
	    cout << endl;
	}
	cout << "sigma_likelihood" << endl;
	for (int i=0; i<sigma_likeli.size(); ++i){
	    for (int j=0; j<sigma_likeli.at(i).size(); ++j){
	        cout << sigma_likeli.at(i).at(j) << "  ";
	    }
	    cout << endl;
	}
	cout << "Exit training" << endl;
}


string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	int x_dim = sample.size();
	int y_dim = possible_labels.size();
    vector<double> class_prob(y_dim);
    
    double norm = 0.0;
    for (int i=0; i<y_dim; ++i){
        double factor = prior.at(i);
        for (int j=0; j<x_dim; ++j){
            factor *= gaussian(sample.at(j), mu_likeli.at(i).at(j), sigma_likeli.at(i).at(j));
        }
        norm += factor;
    }
    
    for (int i=0; i<y_dim; ++i){
        class_prob.at(i) = prior.at(i) / norm;
        for (int j=0; j<x_dim; ++j){
            class_prob.at(i) *= gaussian(sample.at(j), mu_likeli.at(i).at(j), sigma_likeli.at(i).at(j));
        }
    }
    
    
    cout << "Class probabilities" << endl;
    double maximum=class_prob.at(0);
    double max_id = 0;
    for (int i=0; i<y_dim; ++i){
        cout << possible_labels.at(i) << "  " << class_prob.at(i) << endl;
        if (class_prob.at(i) > maximum) {
            maximum = class_prob.at(i);
            max_id = i;
        }
    }
    cout << endl;

    return possible_labels.at(max_id);

}

double GNB::gaussian(double x, double mu, double sigma){
    return (1 / sqrt(2*M_PI*sigma)) * exp(-0.5/sigma*pow((x-mu), 2.0));
}