package com.example.cameraxtest;

import android.graphics.Color;

public abstract class Blender
{
    public static class Pixel
    {
        public Pixel(int color)
        {
            r = Color.red(color);
            g = Color.green(color);
            b = Color.blue(color);
            a = Color.alpha(color);
        }
        public int r,g,b,a;
        public int color()
        {
            return Color.argb(a, r, g, b);
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
        double ab = bottom.a / 255.0;
        double as = top.a / 255.0;
        double a2 = ab * (1 - as);
        bottom.r = (int) ((top.r * as + bottom.r * a2)/(as + a2));
        bottom.g = (int) ((top.g * as + bottom.g * a2)/(as + a2));
        bottom.b = (int) ((top.b * as + bottom.b * a2)/(as + a2));
        bottom.a = (int) ((as + a2) * 255);
        if(bottom.r > 255) bottom.r = 255;
        if(bottom.g > 255) bottom.g = 255;
        if(bottom.b > 255) bottom.b = 255;
        if(bottom.a > 255) bottom.a = 255;
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
        double ab = bottom.a / 255.0;
        bottom.r = (int) ((1 - ab) * top.r + ab * bottom.r * top.r / 255);
        bottom.g = (int) ((1 - ab) * top.r + ab * bottom.g * top.g / 255);
        bottom.b = (int) ((1 - ab) * top.r + ab * bottom.b * top.b / 255);
        if(bottom.r > 255) bottom.r = 255;
        if(bottom.g > 255) bottom.g = 255;
        if(bottom.b > 255) bottom.b = 255;
        return bottom.color();
    }
}