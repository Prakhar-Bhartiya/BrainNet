package com.example.livebrain;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;

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


        btnFetchUsers.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                RequestQueue requestQueue = Volley.newRequestQueue(MainActivity.this);
                String url = "http://10.0.2.2:5000/api/test";

                //taken from https://stackoverflow.com/questions/50197554/is-it-possible-to-download-any-file-pdf-or-zip-using-volley-on-android
                InputStreamVolleyRequest fetchUserRequest = new InputStreamVolleyRequest(Request.Method.POST, url, new Response.Listener<byte[]>() {
                    @Override
                    public void onResponse(byte[] response) {
                        try {
                            if(response != null) {
                                File file = new File(appDir,"test.pkl");
                                if(!file.exists()) {
                                    file.createNewFile();
                                } else {
                                    file.delete(); //delete old copy
                                }
                                FileOutputStream outputStream = new FileOutputStream(file);
                                outputStream.write(response);
                                outputStream.close();
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        error.printStackTrace();
                    }
                }, null) {
                    @Override
                    public Map<String, String> getHeaders() throws AuthFailureError {
                        Map<String, String> params = new HashMap<String, String>();
                        params.put("TestKey", "Test");
                        params.put("Author", "Your mom");
                        params.putAll(super.getHeaders());
                        return params;
                    }
                };

                requestQueue.add(fetchUserRequest);

            }
        });


        if(!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();

    }
}