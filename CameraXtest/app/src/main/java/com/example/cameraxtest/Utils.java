package com.example.cameraxtest;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;

import com.example.cameraxtest.CameraImageGraphic.Layer;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;

import java.util.Arrays;
import java.util.List;

public abstract class Utils {
    //rotate bitmap by a specified angle using a matrix
    public static Bitmap RotateBitmap(Bitmap source, float angle)
    {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    //blend layers together using specified blender
    public static Layer blend(Layer bottom, Layer top)
    {
        Blender blender = null;
        switch (top.Type)
        {
            case Normal:
                blender = new NormalBlender();
                break;
            case Multiply:
                blender = new MultiplyBlender();
                break;
            case Screen:
                blender = new ScreenBlender();
                break;
            case HardLight:
                blender = new HardLightBlender();
                break;
            case Overlay:
                blender = new OverlayBlender();
                break;
        }
        Layer result = new Layer(bottom.bitmap.copy(bottom.bitmap.getConfig(), true), bottom.Type);
        for (int x = 0; x < bottom.bitmap.getWidth(); x++) {
            for (int y = 0; y < bottom.bitmap.getHeight(); y++) {
                result.bitmap.setPixel(x, y, blender.blend(bottom.bitmap.getPixel(x, y),
                                                            top.bitmap.getPixel(x, y)));
            }
        }
        return result;
    }

    public static int[] calcHist(Bitmap input)
    {
        int[] hist = new int[256];
        int px;
        for (int x = 0; x < input.getWidth(); x++) {
            for (int y = 0; y < input.getHeight(); y++) {
                px = input.getPixel(x, y);
                px = (Color.red(px) + Color.green(px) + Color.blue(px))/3;
                hist[px]++;
            }
        }
        return hist;
    }

    public static int[] prefixSum(int[] input)
    {
        int[] copy = new int[input.length];
        copy[0] = input[0];
        for (int i = 1; i < input.length; i++) {
            copy[i] += input[i-1];
        }
        return copy;
    }

    public static int checkBrightness(Bitmap input)
    {
        int[] hist = Utils.prefixSum(Utils.calcHist(input));
        if(hist[50]>hist[255]*9/10.0)
        {
            return -1;
        }
        if(hist[255]-hist[204]>hist[255]*9/10.0)
        {
            return 1;
        }
        return 0;
    }

    public static int brightenColor(int color, float ScaleFactor)
    {
        int r,g,b;
        r =(int) Math.min(Color.red(color) * ScaleFactor, 255);
        r = Math.max(r, 200);
        g =(int) Math.min(Color.green(color) * ScaleFactor, 255);
        g = Math.max(g, 200);
        b =(int) Math.min(Color.blue(color) * ScaleFactor, 255);
        b = Math.max(b, 200);
        return Color.argb(255, r, g, b);
    }

    public static double calculateDistance(float x1, float y1, float x2, float y2)
    {
        return Math.sqrt(Math.pow(x1-x2,2) + Math.pow(y1-y2,2));
    }

    public static PointF getEyebrowCenter(List<PointF> top, List<PointF> bottom)
    {
        Path brow = new Path();
        PointF it = top.get(0);
        brow.moveTo(it.x, it.y);
        for (PointF point: top) {
            brow.lineTo(point.x,point.y);
        }
        for(int i=bottom.size()-1; i>=0; --i)
        {
            it = bottom.get(i);
            brow.lineTo(it.x, it.y);
        }
        RectF boundingBox = new RectF();
        brow.computeBounds(boundingBox, true);
        return new PointF(boundingBox.centerX(), boundingBox.centerY());
    }
}
