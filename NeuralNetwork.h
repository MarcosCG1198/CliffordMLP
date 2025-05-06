//////////////////////////////////////////////////////////////////////////////////////////////
// NeuralNetwork.h: implementation of the NeuralNetwork class.
// Author:Julio Cesar Zamora Esquivel
// julio.c.zamora.esquivel@intel.com
// Group:AVA/ACL/SSR/Intel Labs/NTG. 
// License:The use of this code of a portion of this code must follow a Gate 3 review process 
////////////////////////////////////////////////////////////////////////////////////////////



#if !defined(QNN)
#define QNN

#pragma once

//#include "GeneralDefinitions.h"
//#include "ComputeLayer.h"
#include <math.h>
#include <vector>
//#include <fstream>

using namespace std;

//#include "aligned_allocator.h"

class NNLayerBase;


// helpful typedef's
typedef std::vector< NNLayerBase* >  VectorLayers;
typedef std::vector<float> fVectorNeuronOutputs;
typedef std::vector<char> cVectorNeuronOutputs;//only for ShNN
typedef std::vector<float> fVectorNeuronSumE;
typedef std::vector<int> iVectorNeuronSamples;
typedef std::vector<vector<float>> fArrayDeltas;


class Result
{
	public:
		Result(){Sel=-1;Confidence=0;}
	int Sel;
	float Confidence;
};

//-----------------------------------------------------------
class NeuralNetwork  
{
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();
	volatile float m_etaLearningRate; 

	
	void fForwardPropagate();
	void fBackwardPropagate();
	float Train(float* In,float* d);
	
	float Evaluate2D(float x,float y);
	Result Evaluate(float* In);


	void Initialize();
	float ComputeError();
	void GlobalAdjust();
	
	VectorLayers m_Layers;
};

class NNLayerBase
{
public:
	// Constructor/destructor
	NNLayerBase() : m_strLabel(""), m_pPrevLayer(NULL),m_pNextLayer(NULL){};
	NNLayerBase( const char *label, NNLayerBase* pPrev = NULL,NNLayerBase* pNext = NULL): m_strLabel(label), m_pPrevLayer(pPrev),m_pNextLayer(pNext) {};
	virtual ~NNLayerBase() {};
	
	// Methods
	virtual void fForwardPropagate()= 0;
	virtual void fBackwardPropagate()= 0;
	virtual void SetInput(float* Data,unsigned int H,unsigned int W)= 0;
	
	//virtual void ComputeError()=0;

	// State
	fVectorNeuronOutputs m_fNeuronOutputs;
	
	
	fArrayDeltas m_BackDeltas;
	

	string m_strLabel;
	NNLayerBase* m_pPrevLayer;
	NNLayerBase* m_pNextLayer;
	int best;
	float Confidence;
	int m_Channels;
	float Error;
	iVectorNeuronSamples m_Samples;
	fVectorNeuronSumE m_SumConf;
	int elements;
protected:

};


class CNeuronBase
{
public:
	
	CNeuronBase(){};
	~CNeuronBase(void){};
	std::vector<float> w;
	
	//Those values useful for training 
	std::vector<float> R;
	
	float out;

	//,dfx,dfy;

	virtual float evaluate(NNLayerBase *pPrevL)=0;
	virtual float Training(NNLayerBase *pPrevL,float E)=0;
	

	int n;//number of imputs
	
	float r;//learning rate
	float sig;
};



// Input layer - only the m_NeuronOutputs matters
class NNInputLayer : public NNLayerBase
{
public:
	
	NNInputLayer(unsigned int Inputs);
	NNInputLayer(unsigned int Inputs,int Channels);
	virtual ~NNInputLayer() {};

	// Methods
	virtual void fForwardPropagate() {};
	virtual void fBackwardPropagate() {};
	virtual void SetInput(float* Data,unsigned int H,unsigned int W);
	
	virtual Result GetOutput(){Result r;return r;};
};

// useful to compute the winner and propagate the error
class NNOutputLayer : public NNLayerBase
{
public:
	
	//NNOutputLayer(NNLayerBase *prevLayer, unsigned int nOutputNeurons);
	NNOutputLayer(NNLayerBase *prevLayer);
		
	virtual ~NNOutputLayer() {};
	bool TrackError;
	fArrayDeltas historyErr;
	// Methods
	virtual void fForwardPropagate();
	virtual void fBackwardPropagate();
	virtual void SetInput(float* Data,unsigned int H,unsigned int W);

	virtual Result GetOutput(){Result r;r.Confidence=Confidence;r.Sel=best;return r;}
};

// Building networks
bool BuildNNetwork(NeuralNetwork &nn, char *path);
bool AddInputLayer(NeuralNetwork &nn,int Inputs);
bool AddInputLayer(NeuralNetwork &nn,int Inputs,int channels);

bool AddFullyConnectedLayer(NeuralNetwork &nn,int Neurons);
bool AddRBFLayer(NeuralNetwork& nn, int Neurons);

bool AddOutputLayer(NeuralNetwork &nn);


#endif // QNN
