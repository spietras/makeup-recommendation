package com.example.makeuprecommendation;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Response;

public class DrawActivity extends AppCompatActivity {

    private static final String TAG = "DrawActivity";
    private FaceDetector detector;
    private GraphicOverlay overlay;
    private CameraImageGraphic picture;
    private Bitmap bmp;
    private static final MediaType MEDIA_TYPE_IMAGE = MediaType.parse("image/jpg");
    private String message;
    private long startTime, stopTime;
    private Lock lock;
    private Boolean fromCamera = false;
    private JSONObject colors = null;
    private Boolean ready = false;

    private final OkHttpClient client = new OkHttpClient();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        Intent intent = getIntent();
        message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        //Log.d(TAG, "onCreate: " + message);
        if(message.startsWith("/"))
        {
            message = "file:" + message.substring(5);
            fromCamera = true;
        }
        if(message.startsWith("file"))
        {
            fromCamera = true;
        }

        Executor webExecutor = Executors.newSingleThreadExecutor();

        lock = new ReentrantLock();
        lock.lock();

        webExecutor.execute(() -> {
            String TAG = "HTTPPOST";
            try {
                Response response = Utils.uploadImage("http://192.168.8.138:8080/", Uri.parse(message), fromCamera);
                if (!response.isSuccessful()) {
                    Log.d(TAG, "http post failure");
                    finish();
                }
                else {
                    Log.d(TAG, "http post success");
                    this.colors = new JSONObject(response.body().string());
                }
            } catch (IOException | JSONException e) {
                e.printStackTrace();
                this.colors = null;
            } finally{
                ready = true;
                lock.lock();
                drawMakeup();
                lock.unlock();
            }
        });

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();

        detector = FaceDetection.getClient(options);
        overlay = findViewById(R.id.graphicOverlay);
        overlay.setLayerType(View.LAYER_TYPE_SOFTWARE, null);
        lock.unlock();
        Button button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(ready) runLivePreview();
                else Toast.makeText(DrawActivity.this, "Proszę czekać, trwa przygotowanie kolorów", Toast.LENGTH_SHORT).show();
            }
        });

    }

    public void drawMakeup()
    {
        //int rotation = (intent.getIntExtra(MainActivity.EXTRA_TYPE, 0) == 0)? 270:1;
        InputImage image = null;
        try {
            image = InputImage.fromFilePath(this, Uri.parse(message));
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Cannot open selected file", Toast.LENGTH_SHORT).show();
            finish();
        }
        bmp = image.getBitmapInternal();

        overlay.clear();
        overlay.setImageSourceInfo(bmp.getWidth(), bmp.getHeight(), true);
        overlay.add(new CameraImageGraphic(overlay, bmp));
        startTime = System.currentTimeMillis();
        //Task<List<Face>> result =
        detector.process(image)
                .addOnSuccessListener(this::onSuccess)
                .addOnFailureListener(this::onFailure);
    }

    public void onSuccess(List<Face> faces) {
        stopTime = System.currentTimeMillis();
        if (faces.isEmpty()) {
            Toast.makeText(DrawActivity.this, "Nie wykryto twarzy na zdjęciu", Toast.LENGTH_SHORT).show();
            return;
        }
        Path dummyShadow;
        Paint lipsPaint = new Paint();
        Paint lipsPaintOver = new Paint();
        Paint eyeshadowPaint = new Paint();
        int[] colors;
        int[] dummyCoords = {16,49,21,42,31,37,45,32,57,28,69,26,87,24,102,26,117,31,130,39,140, 50,145,61,148,68,128,74,112,82,92,87,75,87,58,83,40,75,25,64,21,60,17,56,16,49};
        dummyShadow = new Path();
        dummyShadow.moveTo(dummyCoords[0],dummyCoords[1]);
        for (int i=2; i<dummyCoords.length; i+=2)
        {
            dummyShadow.lineTo(dummyCoords[i],dummyCoords[i+1]);
        }
        dummyShadow.close();
        for (Face face : faces) {
            if(this.colors == null) {
                Toast.makeText(this, "Połączenie z serwerem nie udało się, zostaną przydzielone kolory zastępcze", Toast.LENGTH_SHORT).show();
                overlay.add(new FaceGraphic(overlay, face));
            }
            else {
                try {
                    JSONArray array = this.colors.getJSONArray("lipstick_color");
                    JSONArray base = this.colors.getJSONArray("lips_color");
                    int brightness = base.getInt(0) + base.getInt(1) + base.getInt(2);
                    /*int r,g,b;
                    r = (int) (base.getDouble(0) * 256 / array.getDouble(0));
                    g = (int) (base.getDouble(1) * 256 / array.getDouble(1));
                    b = (int) (base.getDouble(2) * 256 / array.getDouble(2));
                    r = Math.min(r, 255);
                    g = Math.min(g, 255);
                    b = Math.min(b, 255);*/
                    /*r = array.getInt(0) - base.getInt(0);
                    g = array.getInt(1) - base.getInt(1);
                    b = array.getInt(2) - base.getInt(2);*/
                    //Log.d(TAG, "base: " + base.toString() + "\nlipstick: " + array.toString() + "\ncrap: " + r + " " + g + " " + b);
                    //lipsPaint.setColor(Color.rgb(r,g,b));
                    lipsPaint.setColor(Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2)));
                    lipsPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
                    lipsPaintOver.setColor(Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2)));
                    lipsPaint.setAlpha(125);
                    lipsPaintOver.setAlpha(50);//50);
                    eyeshadowPaint.setAlpha(80);
                    eyeshadowPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
                    colors = new int[3];
                    array = this.colors.getJSONArray("eyeshadow_outer_color");
                    colors[0] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
                    array = this.colors.getJSONArray("eyeshadow_middle_color");
                    colors[1] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
                    array = this.colors.getJSONArray("eyeshadow_inner_color");
                    colors[2] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
                }
                catch(JSONException e){
                    e.printStackTrace();
                    return;
                }
                overlay.add((new FastGraphic(overlay, face, lipsPaint, lipsPaintOver, eyeshadowPaint, dummyShadow, colors)));
                Log.d(TAG, "onSuccess: Makeup applied");
            }
        }
        overlay.postInvalidate();
    }

    public void onFailure(@NonNull Exception e) {
            stopTime = System.currentTimeMillis();
            e.printStackTrace();
            Toast.makeText(this, "Failed to detect faces", Toast.LENGTH_SHORT).show();
        }

    @Override
    public void onDestroy()
    {
        //Log.d(TAG, "onDestroy: Detection took " + (stopTime - startTime) + " milliseconds");
        super.onDestroy();
        detector.close();
    }

    public void runLivePreview()
    {
        Intent intent = new Intent(this, LivePreviewActivity.class);
        intent.putExtra("colors", this.colors == null?null:this.colors.toString());
        startActivity(intent);
    }
}
