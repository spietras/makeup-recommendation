package com.example.cameraxtest;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

//import com.example.cameraxtest.CameraImageGraphic.BlendModes;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

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

    private final OkHttpClient client = new OkHttpClient();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        Intent intent = getIntent();
        message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        if(message.startsWith("/"))
        {
            message = "file:" + message.substring(5);
        }

        Executor webExecutor = Executors.newSingleThreadExecutor();

        lock = new ReentrantLock();
        lock.lock();

        webExecutor.execute(() -> {
            String TAG = "HTTPPOST";
            /*try {
                Response response = Utils.uploadImage("http://192.168.1.104:8080/", Uri.parse(message));
                if (!response.isSuccessful()) {
                    Log.d(TAG, "http post failure");
                    finish();
                }
                else {
                    Log.d(TAG, "http post success");
                    JSONObject jsonObject = new JSONObject(response.body().string());
                    Log.d(TAG, "lips_color returned " + jsonObject.getString("lips_color"));
                    lock.lock();
                    drawMakeup();
                    lock.unlock();
                }
            } catch (IOException | JSONException e) {
                e.printStackTrace();
                finish();
            }*/
            lock.lock();
            drawMakeup();
            lock.unlock();
        });

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();

        detector = FaceDetection.getClient(options);
        //Intent intent = getIntent();
        //String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        overlay = findViewById(R.id.graphicOverlay);
        //picture = new CameraImageGraphic(overlay);
        overlay.setLayerType(View.LAYER_TYPE_SOFTWARE, null);
        lock.unlock();
        Button button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runLivePreview();
            }
        });

        /*overlay.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                overlay.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                overlay.setLayoutParams(new ConstraintLayout.LayoutParams(overlay.getWidth(), overlay.getWidth()*overlay.getImageHeight()/overlay.getImageWidth()));
            }
        });*/
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

        /*bmp = null;
        try {
            bmp = Utils.RotateBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), Uri.parse(message)), rotation);
        } catch (IOException e) {
            Toast.makeText(this, "Image not found!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            return;
        }*/

        /*switch (Utils.checkBrightness(bmp)) {
            case -1:
                Toast.makeText(DrawActivity.this, "Zdjęcie zbyt ciemne", Toast.LENGTH_SHORT).show();
                finish();
                break;
            case 1:
                Toast.makeText(DrawActivity.this, "Zdjęcie zbyt jasne", Toast.LENGTH_SHORT).show();
                finish();
                break;
            case 0:
                break;
        }*/
        //BitmapFactory.Options opts = new BitmapFactory.Options();
        //opts.inSampleSize = 2;

        //String path = message.substring(5);
        //InputImage image = InputImage.fromBitmap(bmp,0);
        //InputImage image = InputImage.fromBitmap(BitmapFactory.decodeFile(path, opts), 270);//bmp, 0);
        overlay.clear();
        overlay.setImageSourceInfo(bmp.getWidth(), bmp.getHeight(), true);
        overlay.add(new CameraImageGraphic(overlay, bmp));
        //picture.clearLayers();
        //picture.addLayer(bmp, BlendModes.Normal);
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
        FaceGraphic graphic = null;
        //Toast.makeText(DrawActivity.this, "Detection took " + (stopTime - startTime) + " milliseconds", Toast.LENGTH_SHORT).show();
        //Face face = faces.get(0);
        //Rect crop;
        for (Face face : faces) {
            /*crop = face.getBoundingBox();
            Log.d(TAG, "left: " + crop.left + " right: " + crop.right + " bottom: " + crop.bottom + " top: " + crop.top);
            if (crop.left < 0) crop.left = 0;
            if (crop.top < 0) crop.top = 0;
            if (crop.right >= bmp.getWidth()) crop.right = bmp.getWidth() - 1;
            if (crop.bottom >= bmp.getHeight()) crop.right = bmp.getHeight() - 1;
            switch (Utils.checkBrightness(Bitmap.createBitmap(bmp, crop.left, crop.top, crop.width(), crop.height()))) {
                case -1:
                    Toast.makeText(DrawActivity.this, "Zdjęcie zbyt ciemne", Toast.LENGTH_SHORT).show();
                    finish();
                    break;
                case 1:
                    Toast.makeText(DrawActivity.this, "Zdjęcie zbyt jasne", Toast.LENGTH_SHORT).show();
                    finish();
                    break;
                case 0:
                    break;
            }*/
            graphic = new FaceGraphic(overlay, face);
            //graphic.face_scale = 2;
            overlay.add(graphic);
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
        startActivity(intent);
    }
}
