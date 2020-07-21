package com.example.cameraxtest;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
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

import java.io.IOException;
import java.util.List;

public class DrawActivity extends AppCompatActivity {

    private FaceDetector detector;
    private GraphicOverlay overlay;
    private CameraImageGraphic picture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //OpenCVLoader.initDebug();
        setContentView(R.layout.activity_draw);
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();

        detector = FaceDetection.getClient(options);
        Intent intent = getIntent();
        String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        overlay = findViewById(R.id.graphicOverlay);
        picture = new CameraImageGraphic(overlay);

        picture.addLayer(BitmapFactory.decodeResource(getResources(), R.drawable.bottom), BlendModes.Normal);
        picture.addLayer(BitmapFactory.decodeResource(getResources(), R.drawable.top3), BlendModes.Multiply);
        overlay.add(picture);
        overlay.postInvalidate();
        /*Bitmap bmp = null;
        try {
            bmp = Utils.RotateBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), Uri.parse(message)), 270);
        } catch (IOException e) {
            Toast.makeText(this, "Image not found!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            return;
        }
        bmp = Utils.equalize(bmp);
        InputImage image = InputImage.fromBitmap(bmp, 0);
        overlay.clear();
        overlay.setImageSourceInfo(bmp.getWidth(), bmp.getHeight(), true);
        overlay.add(picture);
        picture.clearLayers();
        picture.addLayer(bmp, BlendModes.Normal);
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
        }); */
    }

    public void onSuccess(List<Face> faces) {
        for (Face face: faces) {
            overlay.add(new FaceGraphic(overlay, face));
        }
        overlay.postInvalidate();
    }

    public void onFailure(@NonNull Exception e) {
            // Task failed with an exception
            // ...
            e.printStackTrace();
            Toast.makeText(this, "Failed to detect faces", Toast.LENGTH_SHORT).show();
        }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        detector.close();
    }
}
