package com.example.cameraxtest;

import android.graphics.Color;

public abstract class Blender
{
    public static class Pixel
    {
        public Pixel(int color)
        {
            r = Color.red(color)/255.0;
            g = Color.green(color)/255.0;
            b = Color.blue(color)/255.0;
            a = Color.alpha(color)/255.0;
        }
        public double r,g,b,a;
        public int color()
        {
            int[] px = new int[4];
            px[0] = (int) (255*a);
            px[1] = (int) (255*r);
            px[2] = (int) (255*g);
            px[3] = (int) (255*b);
            for(int i=0; i<3; ++i)
            {
                if( px[i]<0   ) px[i] = 0;
                if( px[i]>255 ) px[i] = 255;
            }
            return Color.argb(px[0], px[1], px[2], px[3]);
        }
    }
    public int blend(int a, int b){
        return b;
    }
}

//normal blend mode
class NormalBlender extends Blender
{
    @Override
    public int blend(int a, int b){
        Pixel bottom = new Pixel(a);
        Pixel top = new Pixel(b);
        double a2 = bottom.a * (1 - top.a);
        bottom.r = (top.r * top.a + bottom.r * a2)/(top.a + a2);
        bottom.g = (top.g * top.a + bottom.g * a2)/(top.a + a2);
        bottom.b = (top.b * top.a + bottom.b * a2)/(top.a + a2);
        bottom.a = top.a + a2;
        return bottom.color();
    }
}

//multiply blend mode
class MultiplyBlender extends Blender
{
    @Override
    public int blend(int a, int b){
        Pixel bottom = new Pixel(a);
        Pixel top = new Pixel(b);
        bottom.r = (1 - bottom.a) * top.r + bottom.a * bottom.r * top.r;
        bottom.g = (1 - bottom.a) * top.r + bottom.a * bottom.g * top.g;
        bottom.b = (1 - bottom.a) * top.r + bottom.a * bottom.b * top.b;
        return bottom.color();
    }
}