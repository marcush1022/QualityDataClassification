//standard includes
#include <iostream>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetworkTrainer.h"

using namespace std;

/*******************************************************************
* constructor
********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer(neuralNetwork *nn) : NN(nn),
																epoch(0),
																learningRate(LEARNING_RATE),
	                                                            momentum(MOMENTUM),
																maxEpochs(MAX_EPOCHS),
																desiredAccuracy(DESIRED_ACCURACY),

																trainingSetAccuracy(0),
																validationSetAccuracy(0),
																generalizationSetAccuracy(0),
																trainingSetMSE(0),
																validationSetMSE(0),
																generalizationSetMSE(0)
{
	//initialize delta weights lists
	//--------------------------------------------------------------------------------------------------------
	deltaInputHidden = new(double*[NN->nInput]);
	for (int i = 0; i < NN->nInput; i++)
	{
		deltaInputHidden[i] = new (double[NN->nHidden]);
		for (int j = 0; j < NN->nHidden; j++) deltaInputHidden[i][j] = 0;
	}

	deltaHiddenOutput = new(double*[NN->nHidden]);
	for (int i = 0; i < NN->nHidden; i++)
	{
		deltaHiddenOutput[i] = new (double[NN->nOutput]);
		for (int j = 0; j < NN->nOutput; j++) deltaHiddenOutput[i][j] = 0;
	}

	//initialize delta bias lists
	//----------------------------------------------------------------------------------------------------------
	deltaBiasHidden = new (double [NN->nHidden]);
	for (int i = 0;i < NN->nHidden;i++)
	{
		deltaBiasHidden[i] = 0;
	}

	deltaBiasOutput = new(double [NN->nOutput]);
	for (int i = 0;i < NN->nOutput;i++)
	{
		deltaBiasOutput[i] = 0;
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new(double[NN->nHidden]);
	for (int i = 0; i < NN->nHidden; i++) hiddenErrorGradients[i] = 0;

	outputErrorGradients = new(double[NN->nOutput]);
	for (int i = 0; i < NN->nOutput; i++) outputErrorGradients[i] = 0;
}//neuralNetworkTrainer


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters(double lR, double m)
{
	learningRate = lR;
	momentum = m;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions(int mEpochs, double dAccuracy)
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;
}
/*******************************************************************
* Enable training logging
********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
	//create log file 
	if (!logFile.is_open())
	{
		logFile.open(filename, ios::out);

		if (logFile.is_open())
		{
			//write log file header
			logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;

			//enable logging
			loggingEnabled = true;

			//resolution setting;
			logResolution = resolution;
			lastEpochLogged = -resolution;
		}
	}
}


/*******************************************************************
* calculate output error gradient
********************************************************************/
double neuralNetworkTrainer::getOutputErrorGradient(double desiredValue, double outputValue)
{
	//return error gradient
	return outputValue * (1 - outputValue) * (desiredValue - outputValue);
}


/*******************************************************************
* calculate input error gradient
********************************************************************/
double neuralNetworkTrainer::getHiddenErrorGradient(int j)
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	for (int k = 0; k < NN->nOutput; k++)
		weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[k];

	//return error gradient
	return NN->hiddenNeurons[j] * (1 - NN->hiddenNeurons[j]) * weightedSum;
}


/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void neuralNetworkTrainer::trainNetwork(trainingDataSet* tSet)
{
	cout << endl << " Neural Network Training Starting: " << endl
		<< "==========================================================================" << endl
		<< " LR: " << learningRate <<", Momentum: "<<momentum<< ", Max Epochs : " << maxEpochs << endl
		<< " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
		<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;
	lastEpochLogged = -logResolution;

	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	//Reapet until epoch equals maxEpoch

	while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs)
	{
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		//cout << "single training at epoch " << epoch << endl;
		//Run a single training epoch (BP)
		runTrainingEpoch(tSet->trainingSet);

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = NN->getSetAccuracy(tSet->generalizationSet);
		generalizationSetMSE = NN->getSetMSE(tSet->generalizationSet);

		//Log Training results
		if (loggingEnabled && logFile.is_open() && (epoch - lastEpochLogged == logResolution))
		{
			logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
			lastEpochLogged = epoch;
		}

		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
		//if (previousTAccuracy != trainingSetAccuracy || previousGAccuracy != generalizationSetAccuracy)
		{
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
		}

		//once training set is complete increment epoch
		epoch++;

	}//end while; training complete;

	 //get validation set accuracy and MSE
	validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
	validationSetMSE = NN->getSetMSE(tSet->validationSet);

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
	logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;

	//out validation accuracy and MSE
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
	cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}//trainNetwork


/*******************************************************************
* Run a single training epoch
********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch(vector<dataEntry*> trainingSet)
{
	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;

	//for every training pattern
	for (int tp = 0; tp < (int)trainingSet.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		NN->feedForward(trainingSet[tp]->pattern);

		/*******************************************************************
		* use the output to calculate the mse and run ABC
		********************************************************************/
		backpropagate(trainingSet[tp]->target);

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
		for (int k = 0; k < NN->nOutput; k++)
		{

			//cout << "NN->outputNeurons[k]= " << NN->outputNeurons[k]<<endl;
			//cout << "NN->clampOutput( NN->outputNeurons[k])= " << NN->clampOutput(NN->outputNeurons[k]) << endl;
			//cout << "trainingSet[tp]->target[k]= " << trainingSet[tp]->target[k] << endl;

			  
			//pattern incorrect if desired and output differ
			if (NN->clampOutput(NN->outputNeurons[k]) != trainingSet[tp]->target[k])
				patternCorrect = false;

			//calculate MSE
			mse += pow((NN->outputNeurons[k] - trainingSet[tp]->target[k]), 2);
		}

		//if pattern is incorrect add to incorrect count
		if (!patternCorrect) incorrectPatterns++;

	}//end for


	 //update training accuracy and MSE
	trainingSetAccuracy = 100 - (incorrectPatterns / trainingSet.size() * 100);
	//cout << "incorrectPatterns= " << incorrectPatterns << endl;
	trainingSetMSE = mse / (NN->nOutput * trainingSet.size());
	//cout << "trainingSetMSE= " << trainingSetMSE << endl;
}//runTrainingEpoch


/*******************************************************************
* Propagate errors back through NN and calculate delta values
********************************************************************/
void neuralNetworkTrainer::backpropagate(double* desiredOutputs)
{
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < NN->nOutput; k++)
	{
		//get error gradient for every output node
		//get gradient = get (x1-x2) * f'(x2)
		outputErrorGradients[k] = getOutputErrorGradient(desiredOutputs[k], NN->outputNeurons[k]);

		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j < NN->nHidden; j++)
		{
			//calculate change in weight 
			deltaHiddenOutput[j][k] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k] + 0.9 * deltaHiddenOutput[j][k];
			//deltaHiddenOutput[j][k] += learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k];
		}
	}

	//modify deltas between input and hidden layers
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < NN->nHidden; j++)
	{
		//get error gradient for every hidden node
		hiddenErrorGradients[j] = getHiddenErrorGradient(j);

		//for all nodes in input layer and bias neuron
		for (int i = 0; i < NN->nInput; i++)
		{
			//calculate change in weight 
			deltaInputHidden[i][j] = learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j] + 0.9 * deltaInputHidden[i][j];
			//deltaInputHidden[i][j] += learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j];
		}
	}

	//modify deltas of hidden bias %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for (int i = 0;i < NN->nHidden;i++)
	{
		deltaBiasHidden[i] = learningRate*hiddenErrorGradients[i] + 0.9*deltaBiasHidden[i];
	}

	//modify deltas of output bias %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for (int i = 0;i < NN->nOutput;i++)
	{
		deltaBiasOutput[i] = learningRate*outputErrorGradients[i]+0.9*deltaBiasOutput[i];
	}

	//if using stochastic learning update the weights immediately
	updateWeightsAndBias();
}//backpropagate


/*******************************************************************
* Update weights and bias using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeightsAndBias()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i < NN->nInput; i++)
	{
		for (int j = 0; j < NN->nHidden; j++)
		{
			//update weight
			NN->wInputHidden[i][j] += deltaInputHidden[i][j];
		}
	}

	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < NN->nHidden; j++)
	{
		for (int k = 0; k < NN->nOutput; k++)
		{
			//update weight
			NN->wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
		}
	}
	
	//updata hidden bias %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for (int i = 0;i < NN->nHidden;i++)
	{
		NN->bHidden[i] += deltaBiasHidden[i];
	}
	//updata output bias %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for (int i = 0;i < NN->nOutput;i++)
	{
		NN->bOutput[i] += deltaBiasOutput[i];
	}
	
	
}


