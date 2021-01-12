package com.example.makeuprecommendation;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.net.Uri;
import android.util.Log;

//import com.example.makeuprecommendation.CameraImageGraphic.Layer;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public abstract class Utils {
    //rotate bitmap by a specified angle using a matrix
    public static Bitmap RotateBitmap(Bitmap source, float angle)
    {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
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

    public static double calculateDistance(PointF first, PointF second)
    {
        return calculateDistance(first.x,first.y,second.x,second.y);
    }

    public static float calculateAngle(float x1, float y1, float x2, float y2)
    {
        return (float) Math.atan2(y2 - y1, x2 - x1);
    }

    public static float calculateAngle(PointF first, PointF second)
    {
        return calculateAngle(first.x, first.y, second.x, second.y);
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

    public static Response uploadImage(String url, Uri imagePath, Boolean fromCamera) throws IOException {//, JSONException {
        OkHttpClient okHttpClient = new OkHttpClient.Builder().connectTimeout(1, TimeUnit.SECONDS).build();
        String path = imagePath.getPath();
        File file;
        Log.d("DRAWACTIVITY", "uploadImage: " + fromCamera.toString());
        if (fromCamera)
        {
            file = new File(path);
        }
        else
        {
            file = new File(path.substring(5));
        }
        String FileType = "image/png";
        if(file.getName().endsWith("jpg")) FileType = "image/jpg";
        RequestBody image = RequestBody.create(file, MediaType.parse(FileType));

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("img", file.getPath(), image)
                .build();
        Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();
        return okHttpClient.newCall(request).execute();
        //JSONObject jsonObject = new JSONObject(response.body().string());
        //return jsonObject.optString("image");
    }
}
