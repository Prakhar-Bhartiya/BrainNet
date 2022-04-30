package com.example.livebrain;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;

import com.android.volley.toolbox.Volley;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.Map;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE}, 99);

        File appDir = new File("/sdcard/livenessApp");
        if(!appDir.exists()) {
            appDir.mkdirs();
        }

        //Connect ui to code
        Button btnFetchUsers = findViewById(R.id.btnFetchUsers);
        Button btnFetchAttack = findViewById(R.id.btnFetchAttack);
        Button btnSelectFeature = findViewById(R.id.btnSelectFeature);
        Button btnRunModels = findViewById(R.id.btnRunModels);
        Spinner featuresSpinner = findViewById(R.id.spinnerFeatures);
        Spinner attackSpinner = findViewById(R.id.spinnerAttack);
        EditText editTextUserSelect = findViewById(R.id.editTextUsers);

        btnFetchAttack.setEnabled(false);
        btnFetchUsers.setEnabled(true);
        btnSelectFeature.setEnabled(false);
        btnRunModels.setEnabled(false);

        editTextUserSelect.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void afterTextChanged(Editable editable) {
                btnFetchUsers.setEnabled(true);
            }
        });

        ArrayAdapter<CharSequence> featuresAdapter = ArrayAdapter.createFromResource(this, R.array.features_array, android.R.layout.simple_spinner_item);
        featuresAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        featuresSpinner.setAdapter(featuresAdapter);
        featuresSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                if(adapterView.getSelectedItemPosition() == 0) {
                    btnSelectFeature.setEnabled(false);
                } else {
                    btnSelectFeature.setEnabled(true);
                    btnSelectFeature.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            featuresSpinner.setEnabled(false);
                            btnSelectFeature.setEnabled(false);

                            //set feature

                        }
                    });
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        ArrayAdapter<CharSequence> attackAdapter = ArrayAdapter.createFromResource(this, R.array.attack_array, android.R.layout.simple_spinner_item);
        attackAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        attackSpinner.setAdapter(attackAdapter);
        attackSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                if(adapterView.getSelectedItemPosition() == 0) {
                    btnFetchAttack.setEnabled(false);
                    btnFetchUsers.setEnabled(false);
                } else {
                    btnFetchAttack.setEnabled(true);
                    btnFetchUsers.setEnabled(false);
                    btnFetchAttack.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            btnFetchAttack.setEnabled(false);
                            attackSpinner.setEnabled(false);
                            btnFetchUsers.setEnabled(false);
                            editTextUserSelect.setEnabled(false);

                            //fetch attack data

                        }
                    });
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        btnRunModels.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                featuresSpinner.setEnabled(true);
                attackSpinner.setEnabled(true);
                editTextUserSelect.setEnabled(true);
            }
        });

        if(!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();
        PyObject test = py.getModule("brain");

//        btnFetchUsers.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                RequestQueue requestQueue = Volley.newRequestQueue(MainActivity.this);
//                String url = "http://10.0.2.2:5000/api/test";
//
//
//                //taken from https://stackoverflow.com/questions/50197554/is-it-possible-to-download-any-file-pdf-or-zip-using-volley-on-android
//                InputStreamVolleyRequest fetchUserRequest = new InputStreamVolleyRequest(Request.Method.POST, url, new Response.Listener<byte[]>() {
//                    @Override
//                    public void onResponse(byte[] response) {
//                        try {
//                            if(response != null) {
//                                File file = new File(appDir,"test");
//                                if(!file.exists()) {
//                                    file.createNewFile();
//                                } else {
//                                    file.delete(); //delete old copy
//                                }
//                                FileOutputStream outputStream = new FileOutputStream(file);
//                                outputStream.write(response);
//                                outputStream.close();
//
//                            }
//                        } catch (Exception e) {
//                            e.printStackTrace();
//                        }
//                    }
//                }, new Response.ErrorListener() {
//                    @Override
//                    public void onErrorResponse(VolleyError error) {
//                        error.printStackTrace();
//                    }
//                }, null) {
//                    @Override
//                    public Map<String, String> getHeaders() throws AuthFailureError {
//                        Map<String, String> params = new HashMap<String, String>();
//                        params.putAll(super.getHeaders());
//                        return params;
//                    }
//                };
//
//                requestQueue.add(fetchUserRequest);
//
//            }
//        });




    }
}