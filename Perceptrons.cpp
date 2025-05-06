////////////////////////////////////////////////////////////////////////////////////////////////
// Perceptrons.cpp: implementation of the NeuralNetwork class.
// Author:Julio Cesar Zamora Esquivel
// julio.c.zamora.esquivel@intel.com
// Group:ISR/HRC Intel Labs  
// Code provided for reference purposes only. The use of this code of a portion
// of this code must follow IL Gate review process
///////////////////////////////////////////////////////////////////////////////////////////////

#include "pch.h"
#include "NeuralNetwork.h"
#include "Perceptrons.h"


using namespace std;


CPerceptron::~CPerceptron(void)
{
}

CPerceptron::CPerceptron(int inputs)
{
	//Weights plus one doe the homogeneous cordinates
	w.resize(inputs+1);
	R.resize(inputs);
	for(int i=0;i<inputs;i++)
	{
		w[i]=0.1-(float)(rand()%100)/500;
		R[i]=0;
	}

	//the las weight is the homogeneous cordinate
	w[inputs]=0.1-(float)(rand()%100)/500;
	r=0.9;
	n=inputs;
	sig=2.9;
}

//---------------------------------------------------------------------
float CPerceptron::evaluate(NNLayerBase *pPrevL)
{

	//remember the input x_i is equal to pPrevL->m_fNeuronOutputs[i]
	//starting with the omogeneous weight

	float sum=w[n];
	
	for(int i=0;i<n;i++)
	{
		suma += (pPrevL->m_fNeuronOutputs[i] * w[i]);
	}

	out=1/(1+exp(-sum/sig));

	return out;
}

void CPerceptron::Clear()
{
	R.clear();
	R.resize(n);
}

//-----------------------------------------------------------------
float CPerceptron::Training(NNLayerBase *pPrevL, float E)
{
	//remember the input x_i is equal to pPrevL->m_fNeuronOutputs[i]
	//sensibility
	float g = E * (out) * (1 - out);

	for (int i = 0; i < n; i++)
	{
		R[i] += g * w[i];
		w[i] += r * g * pPrevL->m_fNeuronOutputs[i];

	}

	//Homogeneous input = 1
	w[n] += r * g;
	
	return out;
}


//----------------------LAYER-----------------------------------------------------------------------------------------------
PerceptronsNNLayer::PerceptronsNNLayer(	NNLayerBase *prevLayer, unsigned int nOutputNeurons) : NNLayerBase("Perceptrons",prevLayer)
{
	//Setting next and previous layer
	m_pPrevLayer = prevLayer;
	prevLayer->m_pNextLayer=this;
	
	//Creating space to allocate neurons
	unsigned int nInputNeurons = prevLayer->m_fNeuronOutputs.size();
	m_fNeuronOutputs.resize(nOutputNeurons);
	mNeurons.resize(nOutputNeurons);
	m_BackDeltas.resize(nOutputNeurons);

	//Creating Perceptron layer
	for(int i=0;i<nOutputNeurons;i++)
	{
		CPerceptron Neuron(nInputNeurons);
		mNeurons[i]=Neuron;
	}
	
}



//---------------------------------------------------------------------
void PerceptronsNNLayer::fForwardPropagate()
{
	for(int i=0;i<m_fNeuronOutputs.size();i++)
	{
		m_fNeuronOutputs[i]=mNeurons[i].evaluate(m_pPrevLayer);
	}
}

//------------------------------------------------------------------
void  PerceptronsNNLayer::fBackwardPropagate()
{
	
	for(int i=0;i<mNeurons.size();i++)
	{
		mNeurons[i].Clear();
		
		for(int k=0;k<m_pNextLayer->m_fNeuronOutputs.size();k++)
		mNeurons[i].Training(m_pPrevLayer,m_pNextLayer->m_BackDeltas[k][i]);
		m_BackDeltas[i]=mNeurons[i].R;
	}
}