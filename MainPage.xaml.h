//
// MainPage.xaml.h
// Declaration of the MainPage class.
//

#pragma once

#include "MainPage.g.h"
#include "NeuralNetwork.h"

namespace CliffordMLP
{
	/// <summary>
	/// An empty page that can be used on its own or navigated to within a Frame.
	/// </summary>
	public ref class MainPage sealed
	{
	public:
		MainPage();
		
	private:
		void Evaluate_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
		void Train_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
		void mCanvas_PointerPressed(Platform::Object^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs^ e);
		void DrawLine();
		float hueToRgb(float p, float q, float t);
		void hslToRgb(float h, float s, float l, int& R, int& G, int& B);
		void EvaluateNN();
		void TrainNN();
		NeuralNetwork* mNewMLP;
		Platform::String^ mClassSelected;
		float Data[300][4];
		float Outs[300][10];
		int m_Total;
		float mErrorGlobal;
		void Img_PointerPressed(Platform::Object^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs^ e);
		void mc1_Checked(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
	};
}
