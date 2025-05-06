//
// MainPage.xaml.cpp
// Implementation of the MainPage class.
//

#include "pch.h"
#include "MainPage.xaml.h"
#include "MemoryBuffer.h"
#include <robuffer.h>
#include <unordered_set>
#include <wincodec.h>
#include <Windows.Graphics.DirectX.Direct3D11.interop.h>
#include <unknwn.h>
#include "robuffer.h"
#include "MemoryBuffer.h"
#include <string>

#include <wrl.h> // For Microsoft::WRL::ComPtr
#include <robuffer.h> // For Windows::Storage::Streams::IBufferByteAccess
#include <windows.storage.streams.h> // For Windows::Storage::Streams::IBuffer

using namespace CliffordMLP;

using namespace Platform;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Controls::Primitives;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Navigation;
using namespace Windows::UI::Xaml::Shapes;
using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Storage::Streams;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

MainPage::MainPage()
{
    m_Total = 0;
    mNewMLP = new  NeuralNetwork();
    BuildNNetwork(*mNewMLP, "");
	InitializeComponent();
}


void CliffordMLP::MainPage::Evaluate_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
 
    EvaluateNN();
}
void CliffordMLP::MainPage::Train_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    TrainNN();
}


void CliffordMLP::MainPage::mCanvas_PointerPressed(Platform::Object^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs^ e)
{
   
}
void  CliffordMLP::MainPage::DrawLine()
{
    Line^ line = ref new Line();
    line->X1 = 50;
    line->Y1 = 50;
    line->X2 = 200;
    line->Y2 = 200;
    line->Stroke = ref new SolidColorBrush(Windows::UI::Colors::Black);
    line->StrokeThickness = 2;

    mCanvas->Children->Append(line);
    
    
}
/** Helper method that converts hue to rgb */
float  CliffordMLP::MainPage::hueToRgb(float p, float q, float t)
{
    if (t < 0.0f)
        t += 1.0f;
    if (t > 1.0f)
        t -= 1.0f;
    if (t < 1.0f / 6.0f)
        return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f)
        return q;
    if (t < 2.0f / 3.0f)
        return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}
void CliffordMLP::MainPage::hslToRgb(float h, float s, float l, int& R, int& G, int& B)
{
    float r, g, b;

    if (s == 0) {
        r = g = b = l; // achromatic
    }
    else {
        float q = l < 0.5f ? l * (1 + s) : l + s - l * s;
        float p = 2 * l - q;
        r = hueToRgb(p, q, h + 1.0f / 3.0f);
        g = hueToRgb(p, q, h);
        b = hueToRgb(p, q, h - 1.0f / 3.0f);
    }
    //int[] rgb = {(int) (r * 255), (int) (g * 255), (int) (b * 255)};
    R = r * 255;
    G = g * 255;
    B = b * 255;

    //temporally removed
    //R= std::min(255,R);
    //G= std::min(255,G);
    //B= std::min(255,B);
    return;
}
void CliffordMLP::MainPage::EvaluateNN()
{



    double width = Img->ActualWidth;
    double height = Img->ActualHeight;
   // int width = 100;
   // int height = 100;
    WriteableBitmap^ bitmap = ref new WriteableBitmap(width, height);
    Img->Source = bitmap;

    // Access the pixel buffer
    IBuffer^ buffer = bitmap->PixelBuffer;
    byte* pixels = nullptr;
    unsigned int capacity = 0;
    Microsoft::WRL::ComPtr<IBufferByteAccess> bufferByteAccess;
    reinterpret_cast<IInspectable*>(buffer)->QueryInterface(IID_PPV_ARGS(&bufferByteAccess));
    bufferByteAccess->Buffer(&pixels);

    int n = 0;
    float inc=2;
    for (float y = -1; y < 0.9; y += inc / height)
    {
        int m = 0;
        
        for (float x = -1; x < 1; x += inc / width)
        {


            //Evaluate Network
            float In[2] = { x,y };
           // Result r = mNewMLP->Evaluate(In);


           // Clifford X = x * e1;
           // Clifford Y = y * e1;
            //Eto21(X);
            //Eto21(Y);
           // Clifford In[2] = { {X},{Y} };
            Result r = mNewMLP->Evaluate(In);

            int R = 0, G = 0, B = 0;
            hslToRgb(r.Sel / 6.0, 0.9, (r.Confidence) / 2.0, R, G, B);
            //color = RGB(R, G, B);



            // Set the color of a specific pixel (e.g., at position (50, 50))
            //int x = 50;
            //int y = 50;
            int pixelIndex = (n * width + m) * 4; // 4 bytes per pixel (BGRA)
            pixels[pixelIndex] = B;     // Blue
            pixels[pixelIndex + 1] = G;   // Green
            pixels[pixelIndex + 2] = R;   // Red
            pixels[pixelIndex + 3] = 255; // Alpha
            m+= inc/2;
        }
        n+= inc/2;
       
    }
    mEvaluateProgress->Value = ((100 * n) / height);
    bitmap->Invalidate();

    // Invalidate the bitmap to update the image
    bitmap->Invalidate();
}

void CliffordMLP::MainPage::TrainNN()
{
    

    double width = Img->ActualWidth-100;
    double height = Img->ActualHeight;


    float dy = height;
    float dx = width;
    for (int j = 0; j < width; j++)
    {
        mErrorGlobal = 0;
        for (int m = 0; m < m_Total; m++)
        {


           // Clifford o = Outs[m][1] * e0;
            float x=Data[m][0];
            float x2=2 * (x) / dx - 1.0;
            float y = Data[m][1];
            float y2=2 * (y) / dy - 1.0;
           // Clifford d[5] = { {Outs[m][0] * e0},{Outs[m][1] * e0},{Outs[m][2] * e0},{Outs[m][3] * e0},{Outs[m][4] * e0} };
            //Clifford X1 = x2 * e1;
            //Clifford X2 = y2 * e1;

           // Clifford In[2] = { {X1},{X2} };


            float d[5] = { {Outs[m][0] },{Outs[m][1]},{Outs[m][2]},{Outs[m][3] },{Outs[m][4]} };
            //Clifford X1 = x2 * e1;
           // Clifford X2 = y2 * e1;

            //Clifford In[2] = { {X1},{X2} };
            float In[2] = { x2,y2 };

            mErrorGlobal += fabs(mNewMLP->Train(In, d));

        }

    }

}

void CliffordMLP::MainPage::Img_PointerPressed(Platform::Object^ sender, Windows::UI::Xaml::Input::PointerRoutedEventArgs^ e)
{
    int i = 0;
    Windows::Foundation::Point point = e->GetCurrentPoint(dynamic_cast<UIElement^>(sender))->Position;
    float x = point.X-100;
    float y = point.Y;

    
    float dy = mCanvas->Height;
    float dx = mCanvas->Width;


    //Data[m_Total][0] = -1;
    //Data[m_Total][1] = 2 * (x) / dx - 1.0;
    //Data[m_Total][2] = 1.0 - 2 * (y) / dy;

    Data[m_Total][0] = x;
    Data[m_Total][1] = y;
    Data[m_Total][2] = 1.0;

    
    
    for (int i = 0; i < 7; i++)Outs[m_Total][i] = 0;
    int sel = 0;
    if (mClassSelected == "C1") { Outs[m_Total][0] = 1; sel = 1; }
    if (mClassSelected == "C2") { Outs[m_Total][1] = 1; sel = 2; }
    if (mClassSelected == "C3") { Outs[m_Total][2] = 1; sel = 3; }
    if (mClassSelected == "C4") { Outs[m_Total][3] = 1; sel = 4; }
    if (mClassSelected == "C5") { Outs[m_Total][4] = 1; sel = 5; }
    
    
    //Determine the color
    COLORREF Colors[10] = { 0,0x0000ff,0x00ffff,0x00ff00,0xffff00,0xff0000,0xff00ff,3500,4000,4500 };
    int Combination = 0;
    //for (int k = 0; k < 7; k++)Combination += Colors[k] * Outs[m_Total][k];
    
    //int R = Combination % 256, G = (Combination / 256) % 256, B = Combination / 65536;
   // Windows::UI::Color Catalog[6] = { Windows::UI::Colors::Black,Windows::UI::Colors::Blue,Windows::UI::Colors::Cyan,Windows::UI::Colors::Green,Windows::UI::Colors::Yellow,Windows::UI::Colors::Red};
    Windows::UI::Color Catalog[6] = { Windows::UI::Colors::Black,Windows::UI::Colors::Red,Windows::UI::Colors::Yellow,Windows::UI::Colors::Green, Windows::UI::Colors::Cyan,Windows::UI::Colors::Blue };


    //Windows::UI::Colors::Black
    Line^ line = ref new Line();
    line->X1 = x-2;
    line->Y1 = y-2;
    line->X2 = x+2;
    line->Y2 = y+2;
    line->Stroke = ref new SolidColorBrush(Catalog[sel]);
    line->StrokeThickness = 2;

    mCanvas->Children->Append(line);
    Line^ line2 = ref new Line();
    line2->X1 = x + 2;
    line2->Y1 = y - 2;
    line2->X2 = x - 2;
    line2->Y2 = y + 2;
    line2->Stroke = line->Stroke;
    line->StrokeThickness = 2;
    mCanvas->Children->Append(line2);

   

    m_Total++;




    // Output the coordinates for debugging
    wchar_t buffer[300];
    swprintf_s(buffer, 300, L"Pointer pressed at: (%f, %f)\n", x, y);
    OutputDebugString(buffer);
}


void CliffordMLP::MainPage::mc1_Checked(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    auto radioButton = dynamic_cast<Windows::UI::Xaml::Controls::RadioButton^>(sender);
    if (radioButton != nullptr)
    {
        //ResultTextBlock->Text = "You selected: " + radioButton->Content->ToString();
        mClassSelected = radioButton->Content->ToString();
    }
}
