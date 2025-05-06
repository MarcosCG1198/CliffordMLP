////////////////////////////////////////////////////////////////////////////////////////////////
// Perceptrons.h: implementation of the NeuralNetwork class.
// Author:Julio Cesar Zamora Esquivel
// julio.c.zamora.esquivel@intel.com
// Group:AVA/ACL/SSR/Intel Labs/NTG. 
// Code provided for reference purposes only. The use of this code of a portion
// of this code must follow IL Gate review process
///////////////////////////////////////////////////////////////////////////////////////////////


#if !defined(PERCEPTRONSH)
#define PERCEPTRONSH

#pragma once

#include <math.h>
#include <vector>
#include <fstream>
using namespace std;

class CPerceptron:public CNeuronBase
{
public:
	CPerceptron(int inputs);
	CPerceptron(){};
	
	~CPerceptron(void);
	float evaluate(NNLayerBase *pPrevL);
	float Training(NNLayerBase *pPrevL,float E);
	void Clear();
};



#include "NeuralNetwork.h"
class PerceptronsNNLayer: public NNLayerBase
{
public:
	

	//Mandatory methods
	PerceptronsNNLayer(	NNLayerBase *prevLayer, unsigned int nOutputNeurons);
	void fForwardPropagate();
	void fBackwardPropagate();
	void SetInput(float* Data,unsigned int H,unsigned int W){};
	
	Result GetOutPut(){};
	virtual ~PerceptronsNNLayer(){};
	
	//Additional
	vector<CPerceptron> mNeurons;
	
	
private:
};



#endif // !defined(PERCEPTRONSH)
