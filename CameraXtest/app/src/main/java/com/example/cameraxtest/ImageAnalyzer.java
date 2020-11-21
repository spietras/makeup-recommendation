package com.example.cameraxtest;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.camera.core.Camera;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.core.app.NotificationCompat;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.nio.ByteBuffer;
import java.util.List;

public class ImageAnalyzer implements ImageAnalysis.Analyzer {
    private GraphicOverlay overlay;
    private FaceDetectorOptions options;
    public FaceDetector detector;
    private String TAG = "ANALYZER";

     public ImageAnalyzer(@NonNull GraphicOverlay o)
    {
        overlay = o;
        options = new FaceDetectorOptions.Builder()
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build();
        detector = FaceDetection.getClient(options);
    }

    @Override
    @androidx.camera.core.ExperimentalGetImage
    public void analyze(@NonNull ImageProxy imageProxy) {
        Log.d("CAMERAX_ANALYZER", "Analyzing frame: "+imageProxy.getWidth()+" x "+ imageProxy.getHeight());
        /*@SuppressLint("UnsafeExperimentalUsageError") Image mediaImage = imageProxy.getImage();
        if(mediaImage == null)
        {
            Log.d(TAG, "analyze: mediaimage is null");
            return;
        }*/
        //Bitmap bmp = BitmapUtils.getBitmap(imageProxy);
        Image mediaImage = imageProxy.getImage();
        InputImage image =
                InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
                //InputImage.fromByteArray(imageProxy.getImage().getPlanes(), imageProxy.getWidth(), imageProxy.getHeight(), 270, InputImage.IMAGE_FORMAT_YUV_420_888);
                //InputImage.fromBitmap(bmp, 0);
        overlay.setImageSourceInfo(imageProxy.getWidth(), imageProxy.getHeight(), true);

        //Task<List<Face>> result =
                detector.process(image)
                        .addOnSuccessListener(
                                new OnSuccessListener<List<Face>>() {
                                    @Override
                                    public void onSuccess(List<Face> faces) {
                                        if(faces.isEmpty())
                                        {
                                            Log.d(TAG, "onSuccess: no faces found");
                                        }
                                        overlay.clear();
                                        //overlay.add(new CameraImageGraphic(overlay, bmp));
                                        for (Face face: faces) {
                                            Log.d(TAG, "analyze: face drawn");
                                            FaceGraphic drawFace = new FaceGraphic(overlay, face);
                                            overlay.add(drawFace);
                                        }
                                        overlay.postInvalidate();
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        Log.d(TAG, "Failure");
                                        // Task failed with an exception
                                        // ...
                                    }
                                });
        // Pass image to an ML Kit Vision API
        // ...
        mediaImage.close();
        imageProxy.close();
    }
}
