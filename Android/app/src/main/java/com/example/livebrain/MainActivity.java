package com.example.livebrain;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Connect ui to code
        Button btnFetchUsers = findViewById(R.id.btnFetchUsers);
        Button btnSelectFeature = findViewById(R.id.btnSelectFeature);
        Button btnRunModels = findViewById(R.id.btnRunModels);
        Spinner featuresSpinner = findViewById(R.id.spinnerFeatures);
        EditText editTextUserSelect = findViewById(R.id.editTextUsers);

        ArrayAdapter<CharSequence> featuresAdapter = ArrayAdapter.createFromResource(this, R.array.features_array, android.R.layout.simple_spinner_item);
        featuresAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        featuresSpinner.setAdapter(featuresAdapter);
        featuresSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });


        if(!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();

    }
}