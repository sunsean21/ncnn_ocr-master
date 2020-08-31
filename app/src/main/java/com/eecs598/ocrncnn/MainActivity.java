// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// This file is written modified from Tencent's example app for ncnn

package com.eecs598.ocrncnn;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.EditText;
import android.widget.CheckBox;

import java.io.FileNotFoundException;

public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;

    private TextView infoResult;
    private ImageView imageView;
    private Bitmap yourSelectedImage = null;
    private EditText numThreads;
    private EditText numIteration;
    private CheckBox useGPU;

    private OcrNcnn ocrncnn = new OcrNcnn();

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        boolean ret_init = ocrncnn.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "ocrncnn Init failed");
        }

        infoResult = (TextView) findViewById(R.id.infoResult);
        imageView = (ImageView) findViewById(R.id.imageView);
        useGPU = (CheckBox) findViewById(R.id.checkbox_gpu);
        numThreads = (EditText) findViewById(R.id.threadsnumber);
        numIteration = (EditText) findViewById(R.id.iterationnumber);

        Button buttonImage = (Button) findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;
                String sit = numIteration.getText().toString();
                int iteration = Integer.parseInt(sit);
                String sth = numThreads.getText().toString();
                int thread = Integer.parseInt(sth);
                String result = ocrncnn.Detect(yourSelectedImage, useGPU.isChecked(), thread, iteration);

                if (result == null)
                {
                    infoResult.setText("detect failed");
                }
                else
                {
                    infoResult.setText(result);
                }
            }
        });

        Button buttonDetectTest = (Button) findViewById(R.id.buttonDetectTest);
        buttonDetectTest.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // if (yourSelectedImage == null)
                //     return;
                String sit = numIteration.getText().toString();
                int iteration = Integer.parseInt(sit);
                String sth = numThreads.getText().toString();
                int thread = Integer.parseInt(sth);

                String result = ocrncnn.Test(useGPU.isChecked(), thread, iteration);

                if (result == null)
                {
                    infoResult.setText("test failed");
                }
                else
                {
                    infoResult.setText(result);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try
            {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    // resize to 227x227
                    yourSelectedImage = Bitmap.createBitmap(rgba);
//                    yourSelectedImage = Bitmap.createScaledBitmap(rgba, 227, 227, false);

                    rgba.recycle();

                    imageView.setImageBitmap(bitmap);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
               || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

}
