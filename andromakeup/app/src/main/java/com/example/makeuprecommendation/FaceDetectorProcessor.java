
/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications copyright (C) 2020 M. Kapuscinski
 */

package com.example.makeuprecommendation;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import androidx.annotation.NonNull;

import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.util.Log;
import android.widget.Toast;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.face.FaceLandmark;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;
import java.util.Locale;

/** Face Detector Demo. */
public class FaceDetectorProcessor extends VisionProcessorBase<List<Face>> {

    private static final String TAG = "FaceDetectorProcessor";

    private final FaceDetector detector;
    private final Path dummyShadow = new Path();
    private final Paint lipsPaint = new Paint();
    private final Paint lipsPaintOver = new Paint();
    private final Paint eyeshadowPaint = new Paint();
    private int[] colors;
    private final JSONObject makeup;

    public void setup() {
        lipsPaint.setColor(Color.rgb(194, 83, 107));
        lipsPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        lipsPaintOver.setColor(Color.rgb(194, 83, 107));
        lipsPaint.setAlpha(125);
        lipsPaintOver.setAlpha(50);
        eyeshadowPaint.setAlpha(80);
        eyeshadowPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        colors = new int[3];
        colors[0] = Color.rgb(143, 15, 58);
        colors[1] = Color.rgb(205, 0, 93);
        colors[2] = Color.rgb(221, 96, 129);
        int[] dummyCoords = {16,49,21,42,31,37,45,32,57,28,69,26,87,24,102,26,117,31,130,39,140, 50,145,61,148,68,128,74,112,82,92,87,75,87,58,83,40,75,25,64,21,60,17,56,16,49};
        dummyShadow.moveTo(dummyCoords[0],dummyCoords[1]);
        for (int i=2; i<dummyCoords.length; i+=2)
        {
            dummyShadow.lineTo(dummyCoords[i],dummyCoords[i+1]);
        }
        dummyShadow.close();
    }

    public void setupJSON() throws JSONException {
        JSONArray array = this.makeup.getJSONArray("lipstick_color");
        JSONArray base = this.makeup.getJSONArray("lips_color");
        int r,g,b;
        r = (int) (base.getDouble(0) * 256 / array.getDouble(0));
        g = (int) (base.getDouble(1) * 256 / array.getDouble(1));
        b = (int) (base.getDouble(2) * 256 / array.getDouble(2));
        r = Math.min(r, 255);
        g = Math.min(g, 255);
        b = Math.min(b, 255);
                    /*r = array.getInt(0) - base.getInt(0);
                    g = array.getInt(1) - base.getInt(1);
                    b = array.getInt(2) - base.getInt(2);*/
        //lipsPaint.setColor(Color.rgb(r,g,b));
        lipsPaint.setColor(Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2)));
        lipsPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        lipsPaintOver.setColor(Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2)));
        lipsPaint.setAlpha(125);
        lipsPaintOver.setAlpha(50);//50);
        eyeshadowPaint.setAlpha(200);
        eyeshadowPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        colors = new int[3];
        array = this.makeup.getJSONArray("eyeshadow_outer_color");
        colors[0] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
        array = this.makeup.getJSONArray("eyeshadow_middle_color");
        colors[1] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
        array = this.makeup.getJSONArray("eyeshadow_inner_color");
        colors[2] = Color.rgb(array.getInt(0), array.getInt(1), array.getInt(2));
        int[] dummyCoords = {16,49,21,42,31,37,45,32,57,28,69,26,87,24,102,26,117,31,130,39,140, 50,145,61,148,68,128,74,112,82,92,87,75,87,58,83,40,75,25,64,21,60,17,56,16,49};
        dummyShadow.moveTo(dummyCoords[0],dummyCoords[1]);
        for (int i=2; i<dummyCoords.length; i+=2)
        {
            dummyShadow.lineTo(dummyCoords[i],dummyCoords[i+1]);
        }
        dummyShadow.close();
    }

    public FaceDetectorProcessor(Context context) {
        this(
                context,
                new FaceDetectorOptions.Builder()
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                        .enableTracking()
                        .build(),
                null);
    }

    public FaceDetectorProcessor(Context context, FaceDetectorOptions options, JSONObject colors) {
        super(context);
        this.makeup = colors;
        if(this.makeup != null) {
            try {
                setupJSON();
            } catch (JSONException e) {
                e.printStackTrace();
                Toast.makeText(context, "Błąd pobrania kolorów", Toast.LENGTH_SHORT).show();
                setup();
            }
        }
        else setup();
        Log.v(MANUAL_TESTING_LOG, "Face detector options: " + options);
        detector = FaceDetection.getClient(options);
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<Face>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(@NonNull List<Face> faces, @NonNull GraphicOverlay graphicOverlay) {
        for (Face face : faces) {
            graphicOverlay.add(new FastGraphic(graphicOverlay, face, lipsPaint, lipsPaintOver, eyeshadowPaint, dummyShadow, colors));
            Log.d(TAG, "onSuccess: Makeup applied");
            //graphicOverlay.add(new FaceGraphic(graphicOverlay, face));
            //logExtrasForTesting(face);
        }
    }

    private static void logExtrasForTesting(Face face) {
        if (face != null) {
            Log.v(MANUAL_TESTING_LOG, "face bounding box: " + face.getBoundingBox().flattenToString());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle X: " + face.getHeadEulerAngleX());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Y: " + face.getHeadEulerAngleY());
            Log.v(MANUAL_TESTING_LOG, "face Euler Angle Z: " + face.getHeadEulerAngleZ());

            // All landmarks
            int[] landMarkTypes =
                    new int[] {
                            FaceLandmark.MOUTH_BOTTOM,
                            FaceLandmark.MOUTH_RIGHT,
                            FaceLandmark.MOUTH_LEFT,
                            FaceLandmark.RIGHT_EYE,
                            FaceLandmark.LEFT_EYE,
                            FaceLandmark.RIGHT_EAR,
                            FaceLandmark.LEFT_EAR,
                            FaceLandmark.RIGHT_CHEEK,
                            FaceLandmark.LEFT_CHEEK,
                            FaceLandmark.NOSE_BASE
                    };
            String[] landMarkTypesStrings =
                    new String[] {
                            "MOUTH_BOTTOM",
                            "MOUTH_RIGHT",
                            "MOUTH_LEFT",
                            "RIGHT_EYE",
                            "LEFT_EYE",
                            "RIGHT_EAR",
                            "LEFT_EAR",
                            "RIGHT_CHEEK",
                            "LEFT_CHEEK",
                            "NOSE_BASE"
                    };
            for (int i = 0; i < landMarkTypes.length; i++) {
                FaceLandmark landmark = face.getLandmark(landMarkTypes[i]);
                if (landmark == null) {
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "No landmark of type: " + landMarkTypesStrings[i] + " has been detected");
                } else {
                    PointF landmarkPosition = landmark.getPosition();
                    String landmarkPositionStr =
                            String.format(Locale.US, "x: %f , y: %f", landmarkPosition.x, landmarkPosition.y);
                    Log.v(
                            MANUAL_TESTING_LOG,
                            "Position for face landmark: "
                                    + landMarkTypesStrings[i]
                                    + " is :"
                                    + landmarkPositionStr);
                }
            }
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face left eye open probability: " + face.getLeftEyeOpenProbability());
            Log.v(
                    MANUAL_TESTING_LOG,
                    "face right eye open probability: " + face.getRightEyeOpenProbability());
            Log.v(MANUAL_TESTING_LOG, "face smiling probability: " + face.getSmilingProbability());
            Log.v(MANUAL_TESTING_LOG, "face tracking id: " + face.getTrackingId());
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }
}
