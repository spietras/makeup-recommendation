package com.example.cameraxtest;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Toast;

import com.androidplot.xy.BarFormatter;
import com.androidplot.xy.SimpleXYSeries;
import com.androidplot.xy.XYPlot;
import com.androidplot.xy.XYSeries;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.Arrays;

public class PlotActivity extends AppCompatActivity {

    private XYPlot plot;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_plot);
        // initialize our XYPlot reference:
        plot = findViewById(R.id.plot);
        Intent intent = getIntent();
        String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        Bitmap bmp = null;
        try {
            bmp = Utils.RotateBitmap(MediaStore.Images.Media.getBitmap(this.getContentResolver(), Uri.parse(message)), 270);
        } catch (IOException e) {
            Toast.makeText(this, "Image not found!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
            return;
        }

        int[] y = Utils.calcHist(bmp);
        Integer[] Y = new Integer[256];
        for (int i = 0; i < 256; i++) {
            Y[i] = y[i];
        }
        XYSeries series = new SimpleXYSeries(Arrays.asList(Y), SimpleXYSeries.ArrayFormat.Y_VALS_ONLY, "Jasność");
        BarFormatter barFormatter = new BarFormatter(Color.BLUE, Color.BLACK);
        plot.addSeries(series, barFormatter);
    }

}
