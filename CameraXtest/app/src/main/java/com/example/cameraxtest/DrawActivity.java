package com.example.cameraxtest;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.InputQueue;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.widget.Toast;
import com.example.cameraxtest.CameraImageGraphic.BlendModes;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.time.Instant;
import java.util.List;
import java.util.Timer;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
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

    private final OkHttpClient client = new OkHttpClient();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw);

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();

        /*FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();*/

        detector = FaceDetection.getClient(options);
        Intent intent = getIntent();
        String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        overlay = findViewById(R.id.graphicOverlay);
        picture = new CameraImageGraphic(overlay);

        bmp = null;
        try {
            bmp = Utils.RotateBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), Uri.parse(message)), 270);
        } catch (IOException e) {
            Toast.makeText(this, "Image not found!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            return;
        }

        switch (Utils.checkBrightness(bmp)) {
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
        }

        InputImage image = InputImage.fromBitmap(bmp, 0);
        overlay.clear();
        overlay.setImageSourceInfo(bmp.getWidth(), bmp.getHeight(), true);
        overlay.add(picture);
        picture.clearLayers();
        picture.addLayer(bmp, BlendModes.Normal);
        startTime = System.currentTimeMillis();
        //Task<List<Face>> result =
        detector.process(image)
                        .addOnSuccessListener(this::onSuccess)
                        .addOnFailureListener(this::onFailure);

        overlay.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                overlay.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                overlay.setLayoutParams(new ConstraintLayout.LayoutParams(overlay.getWidth(), overlay.getWidth()*overlay.getImageHeight()/overlay.getImageWidth()));
            }
        });
    }

    public void onSuccess(List<Face> faces) {
        stopTime = System.currentTimeMillis();
        if (faces.isEmpty()) {
            Toast.makeText(DrawActivity.this, "Nie wykryto twarzy na zdjęciu", Toast.LENGTH_SHORT).show();
            return;
        }
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

            overlay.add(new FaceGraphic(overlay, face));
        }
        overlay.postInvalidate();
        /*File file = new File(Uri.parse(message));

        Request request = new Request.Builder()
                    .url("localhost:port")
                    .post(RequestBody.create(MEDIA_TYPE_IMAGE, file))
                    .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful())
                Log.d(TAG, "onSuccess: http post failure");
            else {
                Log.d(TAG, "onSuccess: http post success");
            }
        }
        catch (IOException e){
            Log.e(TAG, "onSuccess: " + e.getMessage(), e);
        }
        finish();*/
    }

    public void onFailure(@NonNull Exception e) {
            stopTime = System.currentTimeMillis();
            // Task failed with an exception
            // ...
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
}
