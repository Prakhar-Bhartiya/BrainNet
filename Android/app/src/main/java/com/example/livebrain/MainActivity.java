package com.example.livebrain;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.chaquo.python.PyException;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();
        PyObject module = py.getModule("plot");

        Button plotBtn = findViewById(R.id.button);
        EditText etX = findViewById(R.id.etX);
        EditText etY = findViewById(R.id.etY);
        ImageView imageOut = findViewById(R.id.imageView);
        plotBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    byte[] output = module.callAttr("plot", etX.getText().toString(), etY.getText().toString()).toJava(byte[].class);
                    Bitmap image = BitmapFactory.decodeByteArray(output, 0, output.length);
                    imageOut.setImageBitmap(image);

                    InputMethodManager keyboard = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                    keyboard.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), 0);
                } catch(PyException e) {
                    Toast.makeText(MainActivity.this, e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        });
    }
}