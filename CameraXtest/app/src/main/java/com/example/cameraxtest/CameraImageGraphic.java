package com.example.cameraxtest;

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
 */

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;

import com.example.cameraxtest.GraphicOverlay.Graphic;

import java.util.ArrayList;
import java.util.List;

/**
 * Draw camera image to background.
 */
public class CameraImageGraphic extends Graphic {

    public enum BlendModes{
        Normal,
        Multiply,
        Screen,
        HardLight,
        Overlay
    }

    public static class Layer {

        public Layer(Bitmap bmp, BlendModes mode){
            bitmap = bmp;
            Type = mode;
        }

        public final Bitmap bitmap;
        public final BlendModes Type;
    }

    private List<Layer> layers;

    public void clearLayers()
    {
        layers.clear();
    }

    public void addLayer(Bitmap bmp, BlendModes mode){
        layers.add(new Layer(bmp, mode));
    }

    public CameraImageGraphic(GraphicOverlay overlay) {
        super(overlay);
        layers = new ArrayList<>();
    }

    @Override
    public void draw(Canvas canvas) {
        if(layers.isEmpty()) return;
        Layer layer = layers.get(layers.size()-1);
        for(int i = layers.size() - 2; i>=0; --i)
        {
            layer = Utils.blend(layers.get(i), layer);
        }
        canvas.drawBitmap(layer.bitmap, getTransformationMatrix(), null);
    }
}