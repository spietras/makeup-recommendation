package com.example.cameraxtest;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import com.example.cameraxtest.CameraImageGraphic.Layer;

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

    public static Bitmap equalize(Bitmap input)
    {
        int[] hist = new int[256];
        return input;
    }
}
