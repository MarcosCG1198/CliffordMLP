////////////////////////////////////////////////////////////////////////////////////////////////
// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
// Author:Julio Cesar Zamora Esquivel
// julio.c.zamora.esquivel@intel.com
// Group:AVA/ACL/SSR/Intel Labs/NTG. 
// License:The use of this code of a portion of this code must follow a Gate 3 review process 
///////////////////////////////////////////////////////////////////////////////////////////////

#include "pch.h"
#include "NeuralNetwork.h"
#include "Perceptrons.h"
#include "RBF.h"




using namespace std;



NeuralNetwork::NeuralNetwork()
{
	Initialize();
}

void NeuralNetwork::Initialize()
{
	// delete all layers
	VectorLayers::iterator it;
	for( it=m_Layers.begin(); it<m_Layers.end(); it++ )
	{
		delete *it;
	}
	
	m_Layers.clear();
}

NeuralNetwork::~NeuralNetwork()
{
	Initialize();
}
//-------------------------------
void NeuralNetwork::fForwardPropagate()
{
	VectorLayers::iterator l = m_Layers.begin();
	//int count = 1;
	for( l++; l<m_Layers.end(); l++ )
	{
		(*l)->fForwardPropagate();
		//count++;
	}

}


//----------------------------------------------------
void NeuralNetwork::fBackwardPropagate()
{
	VectorLayers::iterator l = m_Layers.end();
	//int count = 1;
	for( l--; l>m_Layers.begin(); l-- )
	{
		(*l)->fBackwardPropagate();
		//count++;
	}

}



//-----------segments layer independent
bool BuildNNetwork(NeuralNetwork &nn, char *path)
{

	AddInputLayer(nn,2);
	
	AddRBFLayer(nn, 5);
	AddFullyConnectedLayer(nn,5);
	
	AddOutputLayer(nn);

	
	return true;
}

//--------------------------------------------------
bool AddInputLayer(NeuralNetwork &nn,int Inputs)
{
	
	//cleaning previous network
	nn.Initialize();

	// Start building the network, creating the input layer 
	NNInputLayer *pInputLayer = new NNInputLayer(Inputs);
	nn.m_Layers.push_back(pInputLayer);
	
	if(pInputLayer!=NULL)return true;
	return false;
}

//--------------------------------------------------
bool AddInputLayer(NeuralNetwork &nn,int Inputs,int channels)
{
	
	//cleaning previous network
	nn.Initialize();

	// Start building the network, creating the input layer 
	NNInputLayer *pInputLayer = new NNInputLayer(Inputs,channels);
	nn.m_Layers.push_back(pInputLayer);
	
	if(pInputLayer!=NULL)return true;
	return false;
}


//-------------------------------------------------------------
bool AddFullyConnectedLayer(NeuralNetwork &nn,int Neurons)
{
	//Final layer
	int size=nn.m_Layers.size();
	PerceptronsNNLayer *pFullyConnectedLayer2 = new PerceptronsNNLayer(nn.m_Layers[size-1],Neurons);
	nn.m_Layers.push_back(pFullyConnectedLayer2);
	return true;
}


bool AddRBFLayer(NeuralNetwork& nn, int Neurons)
{
	//Final layer
	int size = nn.m_Layers.size();
	RBFNNLayer* RBFFullyConnectedLayer2 = new RBFNNLayer(nn.m_Layers[size - 1], Neurons);
	nn.m_Layers.push_back(RBFFullyConnectedLayer2);

	return true;
}





//-------------------------------------------------------------
bool AddOutputLayer(NeuralNetwork &nn)
{
	//The output
	int size=nn.m_Layers.size();
	NNOutputLayer *pOutputLayer = new NNOutputLayer(nn.m_Layers[size-1]);
	nn.m_Layers.push_back(pOutputLayer);
	return true;
}


//----------------------------------------------------------------------
void NNInputLayer::SetInput(float* Data,unsigned int H,unsigned int W)
{
	
	for(int i=0;i<H*W;i++)
	{
		m_fNeuronOutputs[i]=Data[i];
		
	}

}


//----------------------------------------------------------------------
void NNOutputLayer::SetInput(float* Data,unsigned int H,unsigned int W)
{
	//int i=0;
	for(int i=0;i<H*W;i++)
	{
		m_fNeuronOutputs[i]=Data[i];	
	}
}

NNInputLayer::NNInputLayer(unsigned int Inputs) : NNLayerBase("Input",NULL)
{
	m_fNeuronOutputs.resize(Inputs);
	
	m_Channels=1.0;
	elements=2;
}
NNInputLayer::NNInputLayer(unsigned int Inputs,int channels) : NNLayerBase("Input",NULL)
{
	m_fNeuronOutputs.resize(Inputs*channels);
	
	m_Channels=channels;
	elements=2;
}

NNOutputLayer::NNOutputLayer(	NNLayerBase *prevLayer) : NNLayerBase("Output",prevLayer)
{
	m_pPrevLayer = prevLayer;
	prevLayer->m_pNextLayer=this;
	// Create space for outputs equal to the number of previous outputs
	unsigned int nInputNeurons = prevLayer->m_fNeuronOutputs.size();
	m_fNeuronOutputs.resize(nInputNeurons);
	
	m_BackDeltas.resize(nInputNeurons);
	
	historyErr.resize(nInputNeurons);
	TrackError=true;
}

void NNOutputLayer::fForwardPropagate()
{
	float Max=0;
	for(int i=0;i<m_pPrevLayer->m_fNeuronOutputs.size();i++)
	{
		if(m_pPrevLayer->m_fNeuronOutputs[i]>Max){Max=m_pPrevLayer->m_fNeuronOutputs[i];best=i;Confidence=Max;}
	}
	
	
};

void NNOutputLayer::fBackwardPropagate()
{
	float Max=-1;
	Error=0;
	for(int i=0;i<m_fNeuronOutputs.size();i++)
	{
		m_BackDeltas[i].clear();
		for(int j=0;j<m_pPrevLayer->m_fNeuronOutputs.size();j++)
		{
			if(i==j)
			{
				float E=m_fNeuronOutputs[i]-m_pPrevLayer->m_fNeuronOutputs[i];
				m_BackDeltas[i].push_back(E);
				Error+=fabs(E);
				//if(TrackError&&i==0)historyErr[i].push_back(E);
			}
			else
			{
				m_BackDeltas[i].push_back(0.0);
			}
		}
	}
	
	
};


float NeuralNetwork::Train(float* In,float* d)
{
	//float In[2]={x,y};
	//setting inputs 
	int Number_inputs=m_Layers[0]->m_fNeuronOutputs.size();
	m_Layers[0]->SetInput(In,Number_inputs,1);

	//setting outputs
	VectorLayers::iterator l = m_Layers.end();l--;
	 int Number_Outputs=(*l)->m_fNeuronOutputs.size();
	(*l)->SetInput(d,Number_Outputs,1);

	//Learning
	fForwardPropagate();
	fBackwardPropagate();
	float R=ComputeError();
	return R;
}

//-------------------------------------------------
float NeuralNetwork::ComputeError()
{
	VectorLayers::iterator l = m_Layers.end();
	 l--;
	 
	 float E=(*l)->Error;
	 

	 return E;
}
//------------------------------------------------
float NeuralNetwork::Evaluate2D(float x,float y)
{
	float In[2]={x,y};

	m_Layers[0]->SetInput(In,2,1);
	fForwardPropagate();
	VectorLayers::iterator l = m_Layers.end();
	 l--;
	 COLORREF Colors[10]={0,0x0000ff,0xffff00,0x00ffff,0x00ff00,0xff0000,0xff00ff,3500,4000,4500};
	 float color=0;
	 if((*l)->best>=0&&(*l)->Confidence>0.5)color=Colors[(*l)->best+1];//*(*l)->Confidence;
	 return color;//(*l)->Confidence;
	// return m_Layers[3]->Confidence;
}

//------------------------------------------------
Result NeuralNetwork::Evaluate(float* In)
{
	
	int Number_inputs=m_Layers[0]->m_fNeuronOutputs.size();
	m_Layers[0]->SetInput(In,Number_inputs,1);
	fForwardPropagate();
	VectorLayers::iterator l = m_Layers.end();
	 l--;
	 
	 
	 Result R;
	 R.Confidence=(*l)->Confidence;
	 R.Sel=(*l)->best;
	 return R;//;
	// return m_Layers[3]->Confidence;
}



void NeuralNetwork::GlobalAdjust()
{
	
	 return;
}