#include <iostream>
using namespace std;
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

class Random {
public:
	Random() {
		srand(static_cast<unsigned int>(time(nullptr)));
	}

	double rngDouble() {
		return
			static_cast<double>(rand()) /
			RAND_MAX;
	}

};

class Util {
public:
	int getLayer(int neuron, int neuronArrangement[], int layers) {
		for (int i = 0; i < layers; i++) {
			int compensation = 0;
			for (int j = i-1; j >= 0; j--) {compensation = compensation + neuronArrangement[j];}
			if (neuron < (neuronArrangement[i] + compensation)) {return i;}
			if (i == layers - 1 && neuron == neuronArrangement[i] + compensation) {return i;}
		}
		cout << "[X] Encountered an error whilst getting a neuron's layer, neuron: " << neuron << endl;
		exit(1);
		return -1;
	}

	void printImage(int j,vector<vector<unsigned char>> images) {
		for (int i = 0; i < 784; i++) {
			if (images[j][i] == 0) {cout << ".";} else {cout << "#";}
			if ((i+1) % 28 == 0) {cout << endl;}
		}
	}

	double sigmoid(double s_k) {
		double result = 1 / (1 + exp(-s_k));
		return result;
	}
};

int main() {
	// 1. Instantiation
	int neuronArrangement[] = {784,128,10};
	int ncount = 0;
	for (int i = 0; i < (sizeof(neuronArrangement) / sizeof(int)); i++) {ncount = ncount + neuronArrangement[i];} 
	int layers = (sizeof(neuronArrangement) / sizeof(int));
	double** weightMatrix = (double**)malloc(ncount * sizeof(double*));
	for (int i = 0; i < ncount; i++) {
		weightMatrix[i] = (double*)malloc(ncount * sizeof(double));
	}
	
	Random rng;
	Util util;
	
	for (int i = 0; i < ncount; i++) {
		for (int j = 0; j < ncount; j++) {
			if (util.getLayer(i,neuronArrangement,layers) == util.getLayer(j,neuronArrangement,layers) - 1) {
				weightMatrix[i][j] = ((rng.rngDouble()) - 0.5)/1;
			}
			if (i == j) {weightMatrix[i][j] = ((rng.rngDouble()) -0.5)/1;}
		}
	}

	cout << "[!] Finished initializing weights" << endl;
	// 2. Loading Dataset
	
	ifstream file("MNIST/train_labels",ios::binary);
	if (!file) {cout << "[X] Failed to open MNIST/train_labels" << endl; exit(1);}
	int magic_number = 0; file.read(reinterpret_cast<char*>(&magic_number),sizeof(magic_number));
	magic_number = be32toh(magic_number);
	int num_items = 0; file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
	num_items = be32toh(num_items);
	
	vector<unsigned char> labels(num_items);
	file.read(reinterpret_cast<char*>(&labels[0]),num_items);
	file.close();
	
	ifstream file2("MNIST/train_images", ios::binary);
	if (!file2) {cout << "[X] Failed to open MNIST/train_images" << endl; exit(1);}
	int magic_numberi = 0; file2.read(reinterpret_cast<char*>(&magic_numberi),sizeof(magic_numberi));
	magic_numberi = be32toh(magic_numberi);
	int num_itemsi = 0; file2.read(reinterpret_cast<char*>(&num_itemsi), sizeof(num_itemsi));
	num_itemsi = be32toh(num_items);
	int num_rows = 0; file2.read(reinterpret_cast<char*>(&num_rows),sizeof(num_rows));
	num_rows = be32toh(num_rows);
	int num_cols = 0; file2.read(reinterpret_cast<char*>(&num_cols),sizeof(num_cols));
	num_cols = be32toh(num_cols);

	vector<vector<unsigned char>> images(60000, vector<unsigned char>(784));
	for (int i = 0; i < 60000; i++) {
		for (int j = 0; j < 784; j++) {
			file2.read(reinterpret_cast<char*>(&images[i][j]),sizeof(char));
		}
	}
	
	file2.close();
	// 3. Feed-Forward & Back-Prop
	// weightMatrix[ncount][ncount]
	// int neuronArrangement[] = {784,1024,1024,1024,1024,10};
	// int layers;
	double activationVector[ncount];
	int batchSize = 20;
	int epochs = 1;
	double learningRate = 0.1;
    double dampening = 0.1;
	double** diffMatrix = (double**)malloc(ncount * sizeof(double*));
	for (int i = 0; i < ncount; i++) {
		diffMatrix[i] = (double*)malloc(ncount * sizeof(double));
	}
    for (int i = 0; i < ncount; i++) {
        for (int j = 0; j < ncount; j++) {
            diffMatrix[i][j] = 0.0;
        }
    }
	

	for (int i = 0; i < 60000*epochs;) {
		vector<vector<double>> errorPerRun(batchSize,vector<double>(10));
		vector<vector<double>> memActivationVector(batchSize,vector<double>(ncount));
		vector<int> networkAccuracy(batchSize);
		for (int j = 0; j < batchSize; j++) {
			vector<unsigned char> input = images[i+j];
			double magnitude = 0;
			for (int k = 0; k < input.size(); k++) {magnitude = magnitude + (input[k]*input[k]);}
			magnitude = sqrt(magnitude);
			// Feed-Forward
			for (int k = 0; k < layers; k++) {
				if (k == 0) {

					for (int l = 0; l < neuronArrangement[k]; l++) {
						activationVector[l] = input[l]/magnitude;
					}
				}
				else {
					for (int l = 0; l < neuronArrangement[k]; l++) {
						int compensation = 0;
						for (int m = k-1; m >= 0; m--) {compensation = compensation + neuronArrangement[m];}
						int neuron = l + compensation;
						double s_k = weightMatrix[neuron][neuron]; // Initialization to bias
						for (int m = 0; m < ncount; m++) {
							if (util.getLayer(m,neuronArrangement,layers) == util.getLayer(neuron,neuronArrangement,layers) - 1 && activationVector[m] != 0) {
								s_k = s_k + (activationVector[m] * weightMatrix[m][neuron]);
							}
						}
						activationVector[neuron] = util.sigmoid(s_k);
					}
				}
				if (k == layers-1) {
					// Error calculation
					double highest_firing = 0;
					for (int l = ncount-1; l >= ncount - neuronArrangement[layers-1]; l--) {
                        if (activationVector[l] > highest_firing) {
							highest_firing = activationVector[l];
						}
					}
					for (int l = ncount-1; l >= ncount - neuronArrangement[layers-1]; l--) { // output neurons
						// Neurons in ascending order: 0,1,2 ... 9
						int imageLabel = static_cast<int>(labels[i+j]);
						int outputNeuron = ncount - 1 - l;
						double dval = 0.0;
						if (outputNeuron == imageLabel) {dval = 1.0;}
						errorPerRun[j][outputNeuron] = dval - activationVector[l];
						for (int m = 0; m < ncount; m++) {
							memActivationVector[j][m] = activationVector[m];
						}
						if (activationVector[l] == highest_firing && dval == 1.0) {
							networkAccuracy[j] = 1;
						} else if (activationVector[l] == highest_firing && dval == 0.0) {networkAccuracy[j] = 0;}
					}
				}
			}
		}
		// Back-Prop
		// Loss calculation
		// vector<vector<double>> errorPerRun(batchSize,vector<double>(10));
		double loss = 0;
		for (int j = 0; j < errorPerRun.size(); j++) {
			double errorSum = 0;
			for (int k = 0; k < errorPerRun[j].size(); k++) {
				errorSum += pow(errorPerRun[j][k],2);
			}
			errorSum = errorSum / 2;
			loss = loss + (errorSum/errorPerRun.size());
		}
		double acc = 0;
		for (int j = 0; j < networkAccuracy.size(); j++) {
			acc += static_cast<double>(networkAccuracy[j])/static_cast<double>(networkAccuracy.size());
		}
		acc = acc*100;
		cout << "Error in batch " << i/batchSize << ": " << loss << " || " << " With an accuracy of: " << acc << "%" << endl;
		// Weight modification
        
		for (int err = 0; err < batchSize; err++) {
			vector<double> sigmas(ncount);
			for (int j = ncount-1; j >= 0; j--) {
				if (util.getLayer(j,neuronArrangement,layers) == layers-1) {
					sigmas[j] = errorPerRun[err][ncount-1-j] * memActivationVector[err][j]*(1-memActivationVector[err][j]);
					//cout << "S for neuron " << j << " in layer " << util.getLayer(j,neuronArrangement,layers) << " = " << sigmas[j] << endl;
				}
				else {
					double sigmaSum = 0;
					int offset = 0;
					for (int k = util.getLayer(j,neuronArrangement,layers)+1; k < layers; k++) {
						offset += neuronArrangement[k];
					}
					
					int  layerTotal = (ncount-offset) + neuronArrangement[util.getLayer(ncount-offset,neuronArrangement,layers)];
					for (int k = ncount-offset; k < layerTotal; k++) {
						sigmaSum += sigmas[k]*weightMatrix[j][k];
					}
					sigmas[j] = memActivationVector[err][j]*(1-memActivationVector[err][j]) * sigmaSum;
					//cout << "S for neuron " << j << " in layer " << util.getLayer(j,neuronArrangement,layers) << " = " << sigmas[j] << endl;
				}
				//this_thread::sleep_for(chrono::milliseconds(700));
			}
			for (int j = 0; j < ncount; j++) {
				for (int k = 0; k < ncount; k++) {
                    if (util.getLayer(j,neuronArrangement,layers)+1 == util.getLayer(k,neuronArrangement,layers)) {
                        double diff = learningRate*memActivationVector[err][j]*sigmas[k];
					    weightMatrix[j][k] += diff + dampening*diffMatrix[j][k];
                        diffMatrix[j][k] = diff;
                    }
				}
			}
		}

		i = i+batchSize;
	}
	return(0);
};
