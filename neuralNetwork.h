/*******************************************************************
* Basic Feed Forward Neural Network Class
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include "dataReader.h"

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
	int nInput, nHidden, nOutput;

	//use ABC or not

	int select;

	//neurons
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons;

	//weights
	double** wInputHidden;
	double** wHiddenOutput;

	//bias
	double* bHidden;
	double* bOutput;

	//Friends
	//--------------------------------------------------------------------------------------------
	friend neuralNetworkTrainer;

	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
	neuralNetwork(int numInput, int numHidden, int numOutput, int select);
	~neuralNetwork();

	//weight operations
	bool loadWeightsAndBias();
	bool saveWeights(char* outputFilename);
	int* feedForwardPattern(double* pattern);
	double getSetAccuracy(std::vector<dataEntry*>& set);
	double getSetMSE(std::vector<dataEntry*>& set);

	//private methods
	//--------------------------------------------------------------------------------------------

private:

	void initializeWeightsAndBias();
	inline double activationFunction(double x);
	inline int clampOutput(double x);
	void feedForward(double* pattern);

};

#endif
