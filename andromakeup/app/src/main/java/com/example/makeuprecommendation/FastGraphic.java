package com.example.makeuprecommendation;

import android.graphics.BlurMaskFilter;
import android.graphics.Canvas;
import android.graphics.LinearGradient;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Shader;
import android.util.Log;

import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;

import java.util.List;

public class FastGraphic extends GraphicOverlay.Graphic {
    private static final String TAG = "FaceGraphic";

    private final float overlay_scale;
    public float face_scale = 1.0f;

    private final Paint lipsPaint;
    private final Paint lipsPaintOver;
    private final Paint eyeshadowPaint;
    private final PointF dummyCenter = new PointF(89.5F,63F);
    private final int dummySize = 100;
    private final int dummyHeight = 73;
    private final Path dummyShadow;
    private final int[] colors;

    private volatile Face face;

    FastGraphic(GraphicOverlay overlay, Face face, Paint lipsPaint, Paint lipsPaintOver, Paint eyeshadowPaint, Path dummyShadow, int[] colors) {
        super(overlay);

        this.face = face;
        overlay_scale = overlay.scaleFactor;
        this.lipsPaint = lipsPaint;
        this.lipsPaintOver = lipsPaintOver;
        this.eyeshadowPaint = eyeshadowPaint;
        this.dummyShadow = new Path();
        this.dummyShadow.addPath(dummyShadow);
        this.colors = colors;
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

        drawLipsSpline(canvas);
        drawEyeshadow(canvas, FaceContour.LEFT_EYE);
        drawEyeshadow(canvas, FaceContour.RIGHT_EYE);
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

        List<PointF> points;
        points = contour.getPoints();
        //Log.d(TAG, "drawEyeshadow: " + points);
        PointF first, top_end, iter;
        first = points.get(0);
        top_end = points.get(8);
        shadow.moveTo(translateX(first.x), translateY(first.y));
        for(PointF it : points)
        {
            shadow.lineTo(translateX(it.x), translateY(it.y));
        }
        shadow.close();

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
        scaleMatrix.setScale(eye*scaleFactor, scaleY,rectF.centerX(),rectF.centerY());
        shadow_big.transform(scaleMatrix);
        scaleMatrix.setRotate((float) Math.toDegrees(-Utils.calculateAngle(first, top_end)),rectF.centerX(),rectF.centerY());
        shadow_big.transform(scaleMatrix);
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
        BlurMaskFilter blur = new BlurMaskFilter(scale/10, BlurMaskFilter.Blur.NORMAL);
        lipsPaint.setMaskFilter(blur);
        lipsPaintOver.setMaskFilter(blur);
        eyeshadowPaint.setMaskFilter(blur);

        //Log.d(TAG, "drawLipsSpline: scale="+scale);
        float delta = (float) Utils.calculateDistance(face.getContour(FaceContour.UPPER_LIP_BOTTOM).getPoints().get(4), face.getContour(FaceContour.LOWER_LIP_TOP).getPoints().get(4));
        //Log.d(TAG, "drawLipsSpline: delta="+delta);
        if(scale/40<delta)
        {
            //Log.d(TAG, "drawLipsSpline: weszło");
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

        Matrix scaleMatrix = new Matrix();
        RectF rectF = new RectF();
        lips.computeBounds(rectF, true);
        scaleMatrix.setScale(0.75f, 0.75f,rectF.centerX(),rectF.centerY());
        //lips.transform(scaleMatrix);
        canvas.drawPath(lips, lipsPaintOver);
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
        contour = face.getContour(FaceContour.UPPER_LIP_TOP);
        if (contour == null) return;
        points = contour.getPoints();
        for(int i=0; i<points.size(); ++i)
        {
            it = points.get(i);
            lip_lower.cubicTo(translateX(prev.x), translateY(prev.y), translateX(it.x), translateY(it.y), translateX(it.x), translateY(it.y));
            prev = it;
        }
        lip_lower.close();
        canvas.drawPath(lip_lower, lipsPaint);
    }

    @Override
    public float scale(float imagePixel) {
        return imagePixel * overlay_scale * face_scale;
    }
}