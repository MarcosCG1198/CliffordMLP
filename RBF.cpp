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
#include "RBF.h"


using namespace std;


CRBF::~CRBF(void)
{
}

CRBF::CRBF(int inputs)
{
	//Weights plus one doe the homogeneous cordinates
	w.resize(inputs);
	R.resize(inputs);
	for(int i=0;i<inputs;i++)
	{
		w[i]=0.1-(float)(rand()%100)/500;
		R[i]=0;
	}

	//the last weight is the homogeneous cordinate
	//w[inputs]=0.1-(float)(rand()%100)/500;
	r=0.9;
	n=inputs;
	sig=2.9;
}

//---------------------------------------------------------------------
float CRBF::evaluate(NNLayerBase *pPrevL)
{

	//remember the input x_i is equal to pPrevL->m_fNeuronOutputs[i]
	//starting with the omogeneous weight


	float sumG;


			sumG = 0;

			for (int i = 0; i < n; i++)
			{

				sumG += ((pPrevL->m_fNeuronOutputs[i] - w[i]) * (pPrevL->m_fNeuronOutputs[i] - w[i]));

			}

			out = exp(-sumG);

	return out;
}

void CRBF::Clear()
{
	R.clear();
	R.resize(n);
}

//-----------------------------------------------------------------
float CRBF::Training(NNLayerBase *pPrevL, float E)
{
	//remember the input x_i is equal to pPrevL->m_fNeuronOutputs[i]
	//sensibility


	for (int i = 0; i < n; i++)
	{
		
		w[i] -= (pPrevL->m_fNeuronOutputs[i] - w[i]) * 2 * out * E * r);

	}
	
	return out;
}


//----------------------LAYER-----------------------------------------------------------------------------------------------
RBFNNLayer::RBFNNLayer(	NNLayerBase *prevLayer, unsigned int nOutputNeurons) : NNLayerBase("RBF",prevLayer)
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
		CRBF Neuron(nInputNeurons);
		mNeurons[i]=Neuron;
	}
	
}



//---------------------------------------------------------------------
void RBFNNLayer::fForwardPropagate()
{
	for(int i=0;i<m_fNeuronOutputs.size();i++)
	{
		m_fNeuronOutputs[i]=mNeurons[i].evaluate(m_pPrevLayer);
	}
}

//------------------------------------------------------------------
void  RBFNNLayer::fBackwardPropagate()
{
	
	for(int i=0;i<mNeurons.size();i++)
	{
		mNeurons[i].Clear();
		float E = 0;


	

		for (int k = 0; k < m_pNextLayer->m_fNeuronOutputs.size(); k++)
		{


				E += m_pNextLayer->m_BackDeltas[k][i];


		}
			
		mNeurons[i].Training(m_pPrevLayer, E);
	}

			
		
		
	}
}