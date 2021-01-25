package com.example.makeuprecommendation;

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

import android.graphics.BlurMaskFilter;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.Log;

import com.example.makeuprecommendation.GraphicOverlay.Graphic;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.face.FaceLandmark;
import com.google.mlkit.vision.face.FaceLandmark.LandmarkType;

import java.util.List;
import java.util.Locale;

/**
 * Graphic instance for rendering face position, contour, and landmarks within the associated
 * graphic overlay view.
 */
public class FaceGraphic extends Graphic {
    private static final String TAG = "FaceGraphic";
    private static final float FACE_POSITION_RADIUS = 4.0f;
    private static final float ID_TEXT_SIZE = 30.0f;
    private static final float ID_Y_OFFSET = 40.0f;
    private static final float ID_X_OFFSET = -40.0f;
    private static final float BOX_STROKE_WIDTH = 5.0f;
    private static final int NUM_COLORS = 10;
    private static final int[][] COLORS = new int[][]{
            // {Text color, background color}
            {Color.BLACK, Color.WHITE},
            {Color.WHITE, Color.MAGENTA},
            {Color.BLACK, Color.LTGRAY},
            {Color.WHITE, Color.RED},
            {Color.WHITE, Color.BLUE},
            {Color.WHITE, Color.DKGRAY},
            {Color.BLACK, Color.CYAN},
            {Color.BLACK, Color.YELLOW},
            {Color.WHITE, Color.BLACK},
            {Color.BLACK, Color.GREEN}
    };

    private float overlay_scale;
    public float face_scale = 1.0f;

    private final Paint facePositionPaint;
    private final Paint[] idPaints;
    private final Paint[] boxPaints;
    private final Paint[] labelPaints;
    private final Paint lipsPaint = new Paint();
    private final Paint lipsPaintOver = new Paint();
    private final Paint eyeshadowPaint = new Paint();
    private int[] colors;
    private Path dummyShadow;
    private PointF dummyCenter = new PointF(89.5F,63F);
    private int dummySize = 100;
    private int dummyHeight = 73;

    private volatile Face face;

    FaceGraphic(GraphicOverlay overlay, Face face) {
        super(overlay);

        this.face = face;
        final int selectedColor = Color.WHITE;

        facePositionPaint = new Paint();
        facePositionPaint.setColor(selectedColor);

        int numColors = COLORS.length;
        idPaints = new Paint[numColors];
        boxPaints = new Paint[numColors];
        labelPaints = new Paint[numColors];
        for (int i = 0; i < numColors; i++) {
            idPaints[i] = new Paint();
            idPaints[i].setColor(COLORS[i][0] /* text color */);
            idPaints[i].setTextSize(ID_TEXT_SIZE);

            boxPaints[i] = new Paint();
            boxPaints[i].setColor(COLORS[i][1] /* background color */);
            boxPaints[i].setStyle(Paint.Style.STROKE);
            boxPaints[i].setStrokeWidth(BOX_STROKE_WIDTH);

            labelPaints[i] = new Paint();
            labelPaints[i].setColor(COLORS[i][1]  /* background color */);
            labelPaints[i].setStyle(Paint.Style.FILL);
        }
        lipsPaint.setColor(Color.rgb(194,83, 107));
        lipsPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        lipsPaintOver.setColor(Color.rgb(194,83, 107));
        lipsPaint.setAlpha(125);
        lipsPaintOver.setAlpha(50);
        eyeshadowPaint.setAlpha(80);
        eyeshadowPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.OVERLAY));
        colors = new int[3];
        colors[0] = Color.rgb(143,15,58);
        colors[1] = Color.rgb(205, 0, 93);
        colors[2] = Color.rgb(221, 96,129);
       /*for(int i=0; i<colors.length; ++i)
        {
            //colors[i] = Utils.brightenColor(colors[i],3);
        }*/
        overlay_scale = overlay.scaleFactor;

        int[] dummyCoords = {16,49,21,42,31,37,45,32,57,28,69,26,87,24,102,26,117,31,130,39,140, 50,145,61,148,68,128,74,112,82,92,87,75,87,58,83,40,75,25,64,21,60,17,56,16,49};
        dummyShadow = new Path();
        dummyShadow.moveTo(dummyCoords[0],dummyCoords[1]);
        for (int i=2; i<dummyCoords.length; i+=2)
        {
            dummyShadow.lineTo(dummyCoords[i],dummyCoords[i+1]);
        }
        dummyShadow.close();
    }

    /**
     * Draws the face annotations for position on the supplied canvas.
     */
    @Override
    public void draw(Canvas canvas) {
        Face face = this.face;
        if (face == null) {
            return;
        }

        // Draws a circle at the position of the detected face, with the face's track id below.
        float x = translateX(face.getBoundingBox().centerX());
        float y = translateY(face.getBoundingBox().centerY());
        //canvas.drawCircle(x, y, FACE_POSITION_RADIUS, facePositionPaint);

        // Calculate positions.
        float left = x - scale(face.getBoundingBox().width() / 2.0f);
        float top = y - scale(face.getBoundingBox().height() / 2.0f);
        float right = x + scale(face.getBoundingBox().width() / 2.0f);
        float bottom = y + scale(face.getBoundingBox().height() / 2.0f);
        float lineHeight = ID_TEXT_SIZE + BOX_STROKE_WIDTH;
        float yLabelOffset = -lineHeight;

        // Decide color based on face ID
        int colorID = (face.getTrackingId() == null)
                ? 0 : Math.abs(face.getTrackingId() % NUM_COLORS);

        // Calculate width and height of label box
        float textWidth = idPaints[colorID].measureText("ID: " + face.getTrackingId());
        if (face.getSmilingProbability() != null) {
            yLabelOffset -= lineHeight;
            textWidth = Math.max(textWidth, idPaints[colorID].measureText(
                    String.format(Locale.US, "Happiness: %.2f", face.getSmilingProbability())));
        }
        if (face.getLeftEyeOpenProbability() != null) {
            yLabelOffset -= lineHeight;
            textWidth = Math.max(textWidth, idPaints[colorID].measureText(
                    String.format(Locale.US, "Left eye: %.2f", face.getLeftEyeOpenProbability())));
        }
        if (face.getRightEyeOpenProbability() != null) {
            yLabelOffset -= lineHeight;
            textWidth = Math.max(textWidth, idPaints[colorID].measureText(
                    String.format(Locale.US, "Right eye: %.2f", face.getLeftEyeOpenProbability())));
        }

        // Draw labels
        /*canvas.drawRect(left - BOX_STROKE_WIDTH,
                top + yLabelOffset,
                left + textWidth + (2 * BOX_STROKE_WIDTH),
                top,
                labelPaints[colorID]);
        yLabelOffset += ID_TEXT_SIZE;
        canvas.drawRect(left, top, right, bottom, boxPaints[colorID]);
        canvas.drawText("ID: " + face.getTrackingId(), left, top + yLabelOffset,
                idPaints[colorID]);
        yLabelOffset += lineHeight;*/

        // Draws all face contours.
        /*for (FaceContour contour : face.getAllContours()) {
            for (PointF point : contour.getPoints()) {
                canvas.drawCircle(
                        translateX(point.x), translateY(point.y), FACE_POSITION_RADIUS, facePositionPaint);
            }
        }*/


        // Draw facial landmarks
        /*drawFaceLandmark(canvas, FaceLandmark.LEFT_EYE);
        drawFaceLandmark(canvas, FaceLandmark.RIGHT_EYE);
        drawFaceLandmark(canvas, FaceLandmark.LEFT_CHEEK);
        drawFaceLandmark(canvas, FaceLandmark.RIGHT_CHEEK);*/
        drawLipsSpline(canvas);
        drawEyeshadow(canvas, FaceContour.LEFT_EYE);
        drawEyeshadow(canvas, FaceContour.RIGHT_EYE);
    }

    private void drawFaceLandmark(Canvas canvas, @LandmarkType int landmarkType) {
        FaceLandmark faceLandmark = face.getLandmark(landmarkType);
        if (faceLandmark != null) {
            canvas.drawCircle(
                    translateX(faceLandmark.getPosition().x),
                    translateY(faceLandmark.getPosition().y),
                    FACE_POSITION_RADIUS,
                    facePositionPaint);
        }
    }

    private void drawEyeshadow(Canvas canvas, int eyeContour)
    {
        if(eyeContour != FaceContour.LEFT_EYE && eyeContour != FaceContour.RIGHT_EYE)
            return;
        FaceContour contour = face.getContour(eyeContour);
        if(contour == null) return;
        int eye;
        if(eyeContour == FaceContour.LEFT_EYE) eye = -1;
        else eye = 1;
        Path shadow = new Path();
        Path shadow_big = new Path();
        shadow_big.setFillType(Path.FillType.EVEN_ODD);
        shadow_big.addPath(dummyShadow);

        //BezierSpline eyeTop = new BezierSpline(9);
        //BezierSpline eyeBottom = new BezierSpline(9);

        List<PointF> points;
        points = contour.getPoints();
        Log.d(TAG, "drawEyeshadow: " + points);
        PointF first, top_end, iter;
        first = points.get(0);
        top_end = points.get(8);
        shadow.moveTo(translateX(first.x), translateY(first.y));
        //shadow_big.moveTo(translateX(first.x), translateY(first.y));

        /*for(int i=0; i<9; ++i)
        {
            iter = points.get(i);
            eyeTop.set(translateX(iter.x), translateY(iter.y));
        }
        for(int i=8; i<16; ++i)
        {
            iter = points.get(i);
            eyeBottom.set(translateX(iter.x), translateY(iter.y));
        }
        eyeBottom.set(translateX(first.x), translateY(first.y));*/

        /*for(int i=1; i<9; ++i)
        {
            iter = points.get(i);
            shadow_big.lineTo(translateX(iter.x), translateY(iter.y));
        }*/
        for(PointF it : points)
        {
            shadow.lineTo(translateX(it.x), translateY(it.y));
        }
        shadow.close();

        /*eyeTop.applyToPath(shadow);
        eyeBottom.applyToPath(shadow);
        eyeTop.applyToPath(shadow_big);*/

        Matrix scaleMatrix = new Matrix();
        RectF rectF = new RectF();
        shadow.computeBounds(rectF, true);

        scaleMatrix.setTranslate(rectF.centerX()-dummyCenter.x, rectF.centerY()-dummyCenter.y);
        shadow_big.transform(scaleMatrix);
        //ODLEGLOSC NIE DX - OBRACANIE
        float scaleFactor = (float) Utils.calculateDistance(translateX(first.x), translateY(first.y),translateX(points.get(8).x), translateY(points.get(8).y))/dummySize;
        if(eye==-1) iter = Utils.getEyebrowCenter(face.getContour(FaceContour.LEFT_EYEBROW_TOP).getPoints(), face.getContour(FaceContour.LEFT_EYEBROW_BOTTOM).getPoints());
        else iter = Utils.getEyebrowCenter(face.getContour(FaceContour.RIGHT_EYEBROW_TOP).getPoints(), face.getContour(FaceContour.RIGHT_EYEBROW_BOTTOM).getPoints());
        float scaleY = (float) Utils.calculateDistance(translateX(iter.x), translateY(iter.y), rectF.centerX(), rectF.centerY())/dummyHeight;
        //Log.d(TAG, "drawEyeshadow: "+scaleY+" "+scaleFactor);
        scaleMatrix.setScale(eye*scaleFactor, scaleY,rectF.centerX(),rectF.centerY());
        shadow_big.transform(scaleMatrix);
        //Log.d(TAG, "drawEyeshadow: euler"+Utils.calculateAngle(first, top_end));
        scaleMatrix.setRotate((float) Math.toDegrees(-Utils.calculateAngle(first, top_end)),rectF.centerX(),rectF.centerY());
        shadow_big.transform(scaleMatrix);
        //eyeBottom.applyToPath(shadow_big);

        /*for(int i=8; i<16; ++i)
        {
            iter = points.get(i);
            shadow_big.lineTo(translateX(iter.x), translateY(iter.y));
        }
        shadow_big.lineTo(translateX(first.x),translateY(first.y));*/
        //shadow_big.close();

        //shadow.computeBounds(rectF, true);
        //scaleMatrix.setScale(1.5f, 1.5f,rectF.centerX(),rectF.centerY());
        //shadow_big.transform(scaleMatrix);
        shadow_big.addPath(shadow);

        float x1,x2;
        if(eyeContour == FaceContour.LEFT_EYE)
        {
            x1 = first.x;
            x2 = top_end.x;
        }
        else
        {
            x1 = top_end.x;
            x2 = first.x;
        }

        eyeshadowPaint.setShader(new LinearGradient(
                translateX(x1), 0,
                translateX(x2), 0,
                colors,
                null,
                Shader.TileMode.MIRROR
        ));
        canvas.drawPath(shadow_big, eyeshadowPaint);
    }

    private void drawLipsSpline(Canvas canvas)
    {
        Path lips = new Path();
        lips.setFillType(Path.FillType.EVEN_ODD);
        BezierSpline lipsTop = new BezierSpline(face.getContour(FaceContour.UPPER_LIP_TOP).getPoints().size());
        BezierSpline lipsBottom = new BezierSpline(face.getContour(FaceContour.LOWER_LIP_BOTTOM).getPoints().size());
        PointF firstBottom, firstTop;
        FaceContour contour = face.getContour(FaceContour.LOWER_LIP_BOTTOM);
        if (contour == null) return;
        List<PointF> points = contour.getPoints();
        firstBottom = points.get(0);
        lips.moveTo(translateX(firstBottom.x), translateY(firstBottom.y));
        int i=0;
        for(PointF it: points)
        {
            lipsBottom.set(i++, translateX(it.x), translateY(it.y));
        }
        lipsBottom.applyToPath(lips);
        contour = face.getContour(FaceContour.UPPER_LIP_TOP);
        if (contour == null) return;
        points = contour.getPoints();
        firstTop = points.get(0);
        lips.lineTo(translateX(firstTop.x), translateY(firstTop.y));
        i=0;
        for(PointF it: points)
        {
            lipsTop.set(i++, translateX(it.x), translateY(it.y));
        }
        lipsTop.applyToPath(lips);
        lips.close();

        float scale = (float) (Utils.calculateDistance(firstBottom,firstTop));
        //BlurMaskFilter blur = new BlurMaskFilter(scale/10, BlurMaskFilter.Blur.NORMAL);
        BlurMaskFilter blur = new BlurMaskFilter(20, BlurMaskFilter.Blur.NORMAL);
        lipsPaint.setMaskFilter(blur);
        lipsPaintOver.setMaskFilter(blur);
        eyeshadowPaint.setMaskFilter(blur);

        Log.d(TAG, "drawLipsSpline: scale="+scale);
        float delta = (float) Utils.calculateDistance(face.getContour(FaceContour.UPPER_LIP_BOTTOM).getPoints().get(4), face.getContour(FaceContour.LOWER_LIP_TOP).getPoints().get(4));
        Log.d(TAG, "drawLipsSpline: delta="+delta);
        if(scale/40<delta)
        {
            Log.d(TAG, "drawLipsSpline: weszło");
            Path lipsInner = new Path();
            contour = face.getContour(FaceContour.UPPER_LIP_BOTTOM);
            points = contour.getPoints();
            firstTop = points.get(0);
            lipsInner.moveTo(translateX(firstTop.x), translateY(firstTop.y));
            for (PointF it : points) {
                lipsInner.lineTo(translateX(it.x), translateY(it.y));
            }
            contour = face.getContour(FaceContour.LOWER_LIP_TOP);
            points = contour.getPoints();
            for (PointF it : points) {
                lipsInner.lineTo(translateX(it.x), translateY(it.y));
            }
            lips.addPath(lipsInner);
        }

        canvas.drawPath(lips, lipsPaint);

        /*Matrix scaleMatrix = new Matrix();
        RectF rectF = new RectF();
        lips.computeBounds(rectF, true);
        scaleMatrix.setScale(0.75f, 0.75f,rectF.centerX(),rectF.centerY());
        //lips.transform(scaleMatrix);
        canvas.drawPath(lips, lipsPaintOver);*/
    }

    private void drawLips(Canvas canvas)
    {
        Path lip_upper = new Path(), lip_lower = new Path();
        lip_upper.setFillType(Path.FillType.EVEN_ODD);
        lip_lower.setFillType(Path.FillType.EVEN_ODD);
        PointF prev, first, it=null;
        // LOWER
        FaceContour contour = face.getContour(FaceContour.LOWER_LIP_BOTTOM);
        if (contour == null) return;
        List<PointF> points = contour.getPoints();
        prev = first = points.get(0);
        lip_lower.moveTo(translateX(first.x), translateY(first.y));
        for(int i=1; i<points.size(); ++i)
        {
            it = points.get(i);
            lip_lower.cubicTo(translateX(prev.x), translateY(prev.y), translateX(it.x), translateY(it.y), translateX(it.x), translateY(it.y));
            prev = it;
        }
        lipsPaint.setMaskFilter(new BlurMaskFilter(Math.abs(it.x-first.x)/10, BlurMaskFilter.Blur.NORMAL));
        /*contour = face.getContour(FaceContour.LOWER_LIP_TOP);
        if (contour == null) return;
        points = contour.getPoints();
        for(int i=points.size()-1; i>=0; --i)
        {
            it = points.get(i);
            lip_lower.cubicTo(translateX(prev.x), translateY(prev.y), translateX(it.x), translateY(it.y), translateX(it.x), translateY(it.y));
            prev = it;
        }
        // UPPER
        /*contour = face.getContour(FaceContour.UPPER_LIP_BOTTOM);
        if (contour == null) return;
        points = contour.getPoints();
        prev = first = points.get(0);
        lip_upper.moveTo(translateX(first.x), translateY(first.y));
        for(int i=1; i<points.size(); ++i)
        {
            it = points.get(i);
            lip_upper.cubicTo(translateX(prev.x), translateY(prev.y), translateX(it.x), translateY(it.y), translateX(it.x), translateY(it.y));
            prev = it;
        }*/
        contour = face.getContour(FaceContour.UPPER_LIP_TOP);
        if (contour == null) return;
        points = contour.getPoints();
        //for(int i=points.size()-1; i>=0; --i)
            for(int i=0; i<points.size(); ++i)
        {
            it = points.get(i);
            lip_lower.cubicTo(translateX(prev.x), translateY(prev.y), translateX(it.x), translateY(it.y), translateX(it.x), translateY(it.y));
            prev = it;
        }
        lip_lower.close(); //lip_upper.close();
        canvas.drawPath(lip_lower, lipsPaint);
        //canvas.drawPath(lip_upper, lipsPaint);

    }

    @Override
    public float scale(float imagePixel) {
        return imagePixel * overlay_scale * face_scale;
    }
}