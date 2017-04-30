
#include <iostream>
#include <ctime>

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

void main()
{
	//seed random number generator
	srand((unsigned int)time(0));

	//create data set reader and load data file
	//16 is number of input data, 3 is the number of target data
	dataReader d;

	int select=0;

	cout << "USE ABC? Y/N: 1/0" << endl;
	cin >> select;

	d.loadDataFile("winequality3.csv", 11, 3);

	d.setCreationApproach(STATIC);

	//create neural network
	neuralNetwork nn(11, 8, 3, select);

	cout << "created FUCKIN network!!!!!!!!!!!!!!!!!!!!" << endl;

	//create neural network trainer 
	neuralNetworkTrainer nT(&nn);
	nT.setTrainingParameters(0.01,0.9);
	nT.setStoppingConditions(150, 90);
	nT.enableLogging("log.csv", 5);

	cout << "TRAINING FUCKIN START!!!!!!!!!!!!!!!!!!!!" << endl;

	//train neural network on data sets
	for (int i = 0; i < d.getNumTrainingSets(); i++)
	{
		nT.trainNetwork(d.getTrainingDataSet());
	}

	//save the weights1
	nn.saveWeights("weights.csv");

	cout << endl << endl << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
}
