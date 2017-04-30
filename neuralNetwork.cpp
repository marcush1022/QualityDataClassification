//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"

using namespace std;

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO, int sel) : nInput(nI), nHidden(nH), nOutput(nO), select(sel)
{
	//create neuron lists
	
	//--------------------------------------------------------------------------------------------------------
	inputNeurons = new(double[nInput]);
	for (int i = 0; i < nInput; i++)
		inputNeurons[i] = 0;

	hiddenNeurons = new(double[nHidden]);
	for (int i = 0; i < nHidden; i++)
		hiddenNeurons[i] = 0;


	outputNeurons = new(double[nOutput]);
	for (int i = 0; i < nOutput; i++)
		outputNeurons[i] = 0;

	/*********************************************************/
	//create weight lists (no bias neuron weights)
	//wInputHidden and wHiddenOutput are 2 dimen arrays
	//wInputHidden[nInput][nHidden]
	//wHiddenOutput[nHidden][nOutput]
	/******************************************************/
	//--------------------------------------------------------------------------------------------------------
	wInputHidden = new(double*[nInput]);
	for (int i = 0; i < nInput; i++)
	{
		wInputHidden[i] = new (double[nHidden]);
		for (int j = 0; j < nHidden; j++)
			wInputHidden[i][j] = 0;
	}

	wHiddenOutput = new(double*[nHidden]);
	for (int i = 0; i < nHidden; i++)
	{
		wHiddenOutput[i] = new (double[nOutput]);
		for (int j = 0; j < nOutput; j++)
			wHiddenOutput[i][j] = 0;
	}

	bHidden = new(double[nHidden]);
	for (int i = 0; i < nHidden; i++)
	{
		//bHidden[i] = new (double[nOutput]);
		bHidden[i] = 0;
	}

	bOutput = new(double[nOutput]);
	for (int i = 0; i < nOutput; i++)
	{
		//bHidden[i] = new (double[nOutput]);
		bOutput[i] = 0;
	}

	//initialize weights
	//--------------------------------------------------------------------------------------------------------

	if (select == 0)
	{
		cout << "selected no ABC, initializeWeights" << endl;
		initializeWeightsAndBias();
	}

	else
	{
		cout << "selected use ABC, loadWeights And Bias" << endl;
		loadWeightsAndBias();
	}
		
}//neuralNetwork


/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	//delete weight storage
	for (int i = 0; i < nInput; i++) delete[] wInputHidden[i];
	delete[] wInputHidden;

	for (int j = 0; j < nHidden; j++) delete[] wHiddenOutput[j];
	delete[] wHiddenOutput;
}
/*******************************************************************
* Load Neuron Weights and bias from file
********************************************************************/
bool neuralNetwork::loadWeightsAndBias()
{
	//open file for reading
	//fstream inputFile;
	//inputFile.open(filename, ios::in);
	string file_path("WeightsAndBias1.txt");
	fstream openfile(file_path.c_str(), fstream::in);

	if (openfile.is_open())
	{
		double weights[112];
		double bias[11];
		string txtline = "";
		string str[200];
		string nums[200];
		int length = 0;

		int i=0, j, m = 0, n = 0;

		//read data
		while (!openfile.eof())
		{
			getline(openfile, txtline);
			str[i] = txtline;
			i++;
			length++;

		}//while 
		
		for (i = 0;i < length;i++)
		{
			if (i >= 0 && i <= 49 || i >= 60 && i <= 89)
				weights[m++] = atof(str[i].c_str());
			else if (i >= 50 && i <= 59 || i >= 90 && i <= 92)
				bias[n++] = atof(str[i].c_str());
		}

		cout << "m= " << m << ", n= " << n << endl;
		//for (int i = 0;i < n;i++)
			//cout << "bias["<<i<<"]= " << bias[i] << endl;

		//check if sufficient weights were loaded

		if (m != (nInput * nHidden + nHidden * nOutput) || n != (nHidden + nOutput))
		{
			cout << endl << "neuralNetwork.cpp: Error - Incorrect number of weights or bias in input file: " << endl;

			//close file
			openfile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;

			for (i = 0; i < nInput; i++)
			{
				for (j = 0; j < nHidden; j++)
				{
					wInputHidden[i][j] = weights[pos++];
				}
			}

			for (i = 0; i < nHidden; i++)
			{
				for (j = 0; j < nOutput; j++)
				{
					wHiddenOutput[i][j] = weights[pos++];
					//cout << "wHiddenOutput[" << i <<j<< "]= " << wHiddenOutput[i][j] << endl;
				}
			}


			pos = 0;
			//set bias
			for (i = 0;i < nHidden;i++)
				bHidden[i] = bias[pos++];
			for (i = 0;i < nOutput;i++)
				bOutput[i] = bias[pos++];


			//print success
			cout << endl << "neuralNetwork.cpp: Neuron weights loaded successfuly" << endl;

			//close file
			openfile.close();

			return true;
		}
	}
	else
	{
		cout << endl << "neuralNetwork.cpp: Error - Weight input file could not be opened: " << endl;
		return false;
	}
}//loadWeightsAndBias


/*******************************************************************
* Save Neuron Weights
********************************************************************/
bool neuralNetwork::saveWeights(char* filename)
{
	//open file for writting
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if (outputFile.is_open())
	{
		outputFile.precision(50);

		//output weights
		for (int i = 0; i < nInput; i++)
		{
			for (int j = 0; j < nHidden; j++)
			{
				outputFile << wInputHidden[i][j] << ",";
			}
		}

		for (int i = 0; i < nHidden; i++)
		{
			for (int j = 0; j < nOutput; j++)
			{
				outputFile << wHiddenOutput[i][j];
				if (i * nOutput + j + 1 != (nHidden + 1) * nOutput)
					outputFile << ",";
			}
		}

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();

		return true;
	}
	else
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}//saveWeights


/*******************************************************************
* Feed pattern through network and return results
********************************************************************/
int* neuralNetwork::feedForwardPattern(double *pattern)
{
	feedForward(pattern);

	//create copy of output results
	int* results = new int[nOutput];
	for (int i = 0; i < nOutput; i++)
		results[i] = clampOutput(outputNeurons[i]);

	return results;
}
/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy(std::vector<dataEntry*>& set)
{
	double incorrectResults = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//set flag to false if desired and output differ
			if (clampOutput(outputNeurons[k]) != set[tp]->target[k])
				correctResult = false;
		}

		//inc training error for a incorrect result
		if (!correctResult) incorrectResults++;

	}//end for

	 //calculate error and return as percentage
	return 100 - (incorrectResults / set.size() * 100);
}//getSetAccuracy


/*******************************************************************
* Return the NN mean squared error on the set
********************************************************************/
double neuralNetwork::getSetMSE(std::vector<dataEntry*>& set)
{
	double mse = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}

	}//end for

	 //calculate error and return as percentage
	return mse / (nOutput * set.size());
}//getSetMSE


/*******************************************************************
* Activation Function
********************************************************************/
inline double neuralNetwork::activationFunction(double x)
{
	//sigmoid function
	return 1 / (1 + exp(-x));
}
/*******************************************************************
* Output Clamping
********************************************************************/
inline int neuralNetwork::clampOutput(double x)
{
	if (x < 0.2) return 0;
	else if (x > 0.8) return 1;
	else return -1;
}
/*******************************************************************
* Feed Forward Operation
* output = neurons value * w + bias
********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{
	//set input neurons to input values
	for (int i = 0; i < nInput; i++)
	{
		inputNeurons[i] = pattern[i];
		//cout << "pattern=" << pattern[i] << endl;
	}
		

	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < nHidden; j++)
	{
		//clear value
		hiddenNeurons[j] = 0;

		//get weighted sum of pattern 
		for (int i = 0; i < nInput; i++)
		{
			hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j] ;
			//cout << "wInputHidden= " << wInputHidden[i][j] << endl;
		}
		//set to result of sigmoid
		hiddenNeurons[j] = activationFunction(hiddenNeurons[j] - bHidden[j]);
		//hiddenNeurons[j] = activationFunction(hiddenNeurons[j]);
	}

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < nOutput; k++)
	{
		//clear value
		outputNeurons[k] = 0;

		//????????????????????????????????????????????????????????
		//get weighted sum of pattern 
		for (int j = 0; j < nHidden; j++)
			outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k] ;

		//set to result of sigmoid
		outputNeurons[k] = activationFunction(outputNeurons[k] - bOutput[k]);
		//outputNeurons[k] = activationFunction(outputNeurons[k]);
	}
}//feedForward


/*******************************************************************
* Initialize randomly Neuron Weights
* The domian of weights is (-1/sqrt(nInput), 1/sqrt(nInput))
********************************************************************/
void neuralNetwork::initializeWeightsAndBias()
{
	//set range
	double rH = 1 / sqrt((double)nInput);
	double rO = 1 / sqrt((double)nHidden);

	//set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i < nInput; i++)
	{
		for (int j = 0; j < nHidden; j++)
		{
			//set weights to random values
			wInputHidden[i][j] = (((double)(rand() % 100) + 1) / 100 * 2 * rH) - rH;
		}
	}

	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i < nHidden; i++)
	{
		for (int j = 0; j < nOutput; j++)
		{
			//set weights to random values
			wHiddenOutput[i][j] = (((double)(rand() % 100) + 1) / 100 * 2 * rO) - rO;
		}
	}

	//set bias of hiddden
	for (int i = 0;i < nHidden;i++)
	{
		bHidden[i] = (((double)(rand() % 100) + 1) / 100 * 2 * rH) - rH;
	}

	//set bias of output
	for (int i = 0;i < nOutput;i++)
	{
		bOutput[i] = (((double)(rand() % 100) + 1) / 100 * 2 * rO) - rO;
	}
}
