#ifndef GPORTH
#define GPORTH



#include <iostream>
#include <fstream>
#include <string>


#include "graph/gdefs.h"


enum GPortDevice {devScreen, devPrinter, devPicture, devPostscript};


/* Note that we always draw using the following coordinate system:

(0,0)--------->(+x,0)
  |
  |
  |
 \/
(+y,0)

 Hence the origin is the top left hand corner, and y goes down rather than up.
 This is typical for drawing to a window. Some systems have other coordinate
 systems (such as Postscript). We make the translation internally.

*/

// A point
class GPoint
{
public:
    GPoint () { SetPoint (0, 0); };
    GPoint (const GPoint &p) { X = p.X; Y = p.Y; };
    GPoint (const int x, const int y) { SetPoint (x, y); };
    virtual int GetX () const { return X; };
    virtual int GetY () const { return Y; };
    virtual void Offset (const int xoff, const int yoff) { X += xoff; Y += yoff; };
    virtual void SetPoint (const int x, const int y) { X = x; Y = y; };
    virtual void SetX (int x) { X = x; };
    virtual void SetY (int y) { Y = y; };

    int	operator== (const GPoint &p) { return (int) ( (X == p.X) && ( Y == p.Y)); };
    int	operator!= (const GPoint &p) { return (int) ( (X != p.X) || ( Y != p.Y)); };
protected:
    int X;
    int Y;
};

// A rectangle
class GRect
{
public:
    GRect () { left = top = right = bottom = 0; };
    GRect (const int l, const int t, const int r, const int b) { SetRect (l, t, r, b); };
    virtual int  GetLeft () const { return left; };
    virtual int  GetTop () const { return top; };
    virtual int  GetRight () const { return right; };
    virtual int  GetBottom () const { return bottom; };
    virtual int  GetWidth () const { return right - left; };
    virtual int  GetHeight () const { return bottom - top; };

    virtual void Inset (const int dx, const int dy) { left += dx; right -= dx; top += dy; bottom -= dy; };
    virtual void Offset (const int dx, const int dy) { left += dx; right += dx; top += dy; bottom += dy; };
    virtual bool PointInRect (GPoint &pt)
    {
    	return (((pt.GetX() >= left) && (pt.GetX() <= right)) &&
        	((pt.GetY() >= top) && (pt.GetY() <= bottom)));
    }

    virtual void SetLeft (const int l) {left = l; };
    virtual void SetTop (const int t) {top = t; };
    virtual void SetRight (const int r) {right = r; };
    virtual void SetBottom (const int b) {bottom = b; };
    virtual void SetRect (const int l, const int t, const int r, const int b)
    	{ left = l; top = t; right = r; bottom = b; };
    virtual void SetRectWH (const int l, const int t, const int w, const int h)
    	{ left = l; top = t; right = l + w; bottom = t + h; };

protected:
    int left, top, right, bottom;
};

// Base class for system specific fonts
class GBaseFont
{
public:
    GBaseFont ();
    virtual ~GBaseFont () {};
    virtual std::string GetName () { return description; };
    virtual std::string GetDescription () { return description; };
    virtual int  GetSize () { return size; };
    virtual bool IsBold () { return bold; };
    virtual bool IsItalic () { return italic; };
private:
    std::string 	description;
    std::string 	name;
    int 		size;
    bool 		bold;
    bool 		italic;
};
typedef GBaseFont *GBaseFontPtr;

typedef GBaseFont GFont ; // for now
typedef GFont *GFontPtr;



// Postscript - just write to file



// Encapsulates the complete graphics system (screen drawing, picture files,
// printing, clipboard).
class GBasePort
{
public:
    GBasePort () { Device = devScreen; PenWidth = 1;};
    virtual ~GBasePort() {};
    virtual void DrawArc (const GPoint &pt, const int radius,
        const double startAngleDegrees, const double endAngleDegrees) = 0;
    virtual void DrawCircle (const GPoint &pt, const int radius) = 0;
    virtual void DrawLine (const int x1, const int y1, const int x2, const int y2) = 0;
    virtual void DrawLinePts (const GPoint &pt1, const GPoint &pt2)
        { DrawLine (pt1.GetX(), pt1.GetY(), pt2.GetX(), pt2.GetY()); };
    virtual void DrawRect (const GRect &r) = 0;
    virtual void DrawText (const int x, const int y, const char *s) = 0;

	// Display
    virtual GPortDevice GetCurrentDevice () { return Device; };
    virtual void GetDisplayRect (GRect &r) { r = DisplayRect; };
    virtual void SetDisplayRect (GRect &r) { DisplayRect = r; };

    // Pen
    virtual int  GetPenWidth () { return PenWidth; };
    virtual void SetPenWidth (int w) { PenWidth = w; };

    // Fonts
    virtual void SetCurrentFont (GBaseFont &font) = 0;

    // Pictures
    virtual void StartPicture (char *pictFileName) = 0;
    virtual void EndPicture () = 0;

    // Groups
    virtual void BeginGroup () = 0;
    virtual void EndGroup () = 0;

    // Printing
    virtual void GetPrintingRect (GRect &r) = 0;

	// Colour
	virtual void SetFillColorRGB (int /*r*/, int /*g*/, int /*b*/) {};


protected:
    // list of fonts
    // printer class
    //pens

    int PenWidth;

    // Device info
    GPortDevice Device;
    GRect	DisplayRect;
};

class GPostscriptPort : public GBasePort
{
public:
    GPostscriptPort ();
    virtual ~GPostscriptPort () {};
    virtual void DrawArc (const GPoint &pt, const int radius,const double startAngleDegrees, const double endAngleDegrees);
    virtual void DrawCircle (const GPoint &pt, const int radius);
    virtual void DrawLine (const int x1, const int y1, const int x2, const int y2);
    virtual void DrawRect (const GRect &r);
    virtual void DrawText (const int x, const int y, const char *text);
    virtual void FillCircle (const GPoint &pt, const int radius);



    // Pen
    virtual void SetPenWidth (int w);


    // Fonts
    virtual void SetCurrentFont (GBaseFont &font);

    // Pictures
    virtual void StartPicture (char *pictFileName);
    virtual void EndPicture ();

    // Groups
    virtual void BeginGroup () {};
    virtual void EndGroup () {};

    // Printing
    virtual void GetPrintingRect (GRect &r);

	virtual void SetFillColorRGB (int r, int g, int b)
	{
		fill_r = (double)r/255.0;
		fill_g = (double)g/255.0;
		fill_b = (double)b/255.0;
	};

protected:
    std::ofstream 		PostscriptStream;
    std::string 	DocumentFonts;

	double fill_r, fill_g, fill_b;
};

extern GBasePort *Port;

#endif
