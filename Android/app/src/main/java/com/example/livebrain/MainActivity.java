package com.example.livebrain;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

        if(!new File(appDir, "Dataset1.mat").exists()) {
            Toast.makeText(this, "Dataset1.mat does not exist.", Toast.LENGTH_LONG).show();
            try {
                wait(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.exit(0);
        }

        //starts python
        if(!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Python py = Python.getInstance();
        PyObject brainModule = py.getModule("android");

        //Connect ui to code
        Button btnFetchUsers = findViewById(R.id.btnFetchUsers);
        Button btnFetchAttack = findViewById(R.id.btnFetchAttack);
        Button btnSelectFeature = findViewById(R.id.btnSelectFeature);
        Button btnRunModels = findViewById(R.id.btnRunModels);
        Spinner featuresSpinner = findViewById(R.id.spinnerFeatures);
        Spinner attackSpinner = findViewById(R.id.spinnerAttack);
        EditText editTextUserSelect = findViewById(R.id.editTextUsers);

        //No submitting empty data
        btnFetchAttack.setEnabled(false);
        btnFetchUsers.setEnabled(false);
        btnSelectFeature.setEnabled(false);
        btnRunModels.setEnabled(false);

        final boolean[] runBools = {false, false, false}; //0 => fetchUser done, 1 => fetchAttack done, 2=> selectFeature done
        final String[] feature = {""}; //selected feature string
        final int[] userBounds = {-2, -1}; //lower and upper use bounds
        final int[] attack = {-1}; //chosen attack -1 => no attack
        final String[] modelNames = {"logReg", "kmeans", "svm", "knn", "scaler", "pca"};
        final boolean[] truthValue = new boolean[1]; //truth value determined by attack selected

        editTextUserSelect.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void afterTextChanged(Editable editable) {
                Pattern intPattern = Pattern.compile("(?:\\d{1,3}+-*\\d{0,3})"); // #-# pattern
                Matcher intMatcher = intPattern.matcher(editable.toString());
                if(editable.toString().equals("")) { //empty box
                    btnFetchUsers.setEnabled(false);
                } else {
                    if(intMatcher.find()) {
                        String[] intStrings = intMatcher.group().split("-"); //fetchs #, or #-# and valids they are between 0-105
                        int lowerUser = -2, higherUser = -1;
                        if(intStrings.length == 2) { //#-# input case
                            higherUser = Integer.parseInt(intStrings[1]);
                            lowerUser = Integer.parseInt(intStrings[0]);
                            if(lowerUser < higherUser && lowerUser > -1 && lowerUser < 106 && higherUser > -1 && higherUser < 106) {
                                btnFetchUsers.setEnabled(true); //valid input turns on the fetch button
                                int finalLowerUser1 = lowerUser;
                                int finalHigherUser = higherUser;
                                btnFetchUsers.setOnClickListener(new View.OnClickListener() {
                                    @Override
                                    public void onClick(View view) {
                                        userBounds[0] = finalLowerUser1;
                                        userBounds[1] = finalHigherUser;
                                        btnFetchUsers.setEnabled(false);

                                        //attack is not used for aggregate predictions
                                        btnFetchAttack.setEnabled(false);
                                        attackSpinner.setEnabled(false);
                                        attackSpinner.setSelection(0);
                                        attack[0] = -1;

                                        //hides keyboard on submit
                                        InputMethodManager keyboard = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                                        if(keyboard.isAcceptingText()) { //close keyboard if it wasnt already closed
                                            keyboard.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), 0);
                                        }

                                        runBools[0] = true;
                                        runBools[2] = true; //ignore attack selection
                                        Log.d("Bools", Arrays.toString(runBools));
                                        if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                            btnRunModels.setEnabled(true);
                                        }
                                    }
                                });
                            } else {
                                btnFetchUsers.setEnabled(false);
                            }
                        } else if(intStrings.length == 1) { //# input case
                            lowerUser = Integer.parseInt(intStrings[0]);
                            if(lowerUser > -1 && lowerUser < 106) {
                                btnFetchUsers.setEnabled(true);
                                int finalLowerUser = lowerUser;
                                btnFetchUsers.setOnClickListener(new View.OnClickListener() {
                                    @Override
                                    public void onClick(View view) {
                                        userBounds[0] = finalLowerUser;
                                        btnFetchUsers.setEnabled(false);

                                        //hides keyboard on submit
                                        InputMethodManager keyboard = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                                        if(keyboard.isAcceptingText()) { //close keyboard if it wasnt already closed
                                            keyboard.hideSoftInputFromWindow(getCurrentFocus().getWindowToken(), 0);
                                        }

                                        runBools[0] = true;
                                        if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                            btnRunModels.setEnabled(true);
                                        }
                                    }
                                });
                            } else {
                                btnFetchUsers.setEnabled(false);

                            }
                        } else {
                            btnFetchUsers.setEnabled(false);
                        }
                    }
                }
            }
        });

        ArrayAdapter<CharSequence> featuresAdapter = ArrayAdapter.createFromResource(this, R.array.features_array, android.R.layout.simple_spinner_item);
        featuresAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        featuresSpinner.setAdapter(featuresAdapter);
        featuresSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                if(adapterView.getSelectedItemPosition() == 0) { //dropdown currently on ------- not a valid feature
                    btnSelectFeature.setEnabled(false);
                } else {
                    btnSelectFeature.setEnabled(true); //dropdown on valid feature
                    btnSelectFeature.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) { //sets feature[0] to the selected feature and turns off selecting
                            featuresSpinner.setEnabled(false);
                            btnSelectFeature.setEnabled(false);
                            feature[0] = (String) adapterView.getSelectedItem();
                            runBools[1] = true;
                            if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                btnRunModels.setEnabled(true);
                            }
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
                btnFetchAttack.setEnabled(true);
                btnFetchAttack.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        btnFetchAttack.setEnabled(false);
                        attackSpinner.setEnabled(false);

                        attack[0] = adapterView.getSelectedItemPosition() - 1;
                        if(attack[0] != -1) {
                            //fetch attack data
                            truthValue[0] = false;
                            String url = "http://10.0.2.2:5000/api/fetch-attack/" + attack[0]; //host lookback interface
                            String fileName = "";
                            if(attack[0] == 0 || attack[0] == 1) { //VAE or GAN attack
                                fileName = "GeneratedAttackVector.mat";
                            } else {
                                fileName = "sampleAttack.mat"; //Sample Data Attack
                            }

                            File file = new File(appDir, fileName);
                            if(!file.exists() || file.length() == 0) { //fetch file if doesnt exist or if its empty
                                try {
                                    if(!file.exists()) { //create if does not exist
                                        file.createNewFile();
                                    }
                                    DownloadTask download = new DownloadTask();
                                    download.execute(fileName, url);
                                    try {
                                        if(download.get()) { //model downloaded successfully
                                            runBools[2] = true;
                                            if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                                btnRunModels.setEnabled(true);
                                            }
                                        } else {
                                            Log.d("Downloading Failed", file.getName());
                                            Toast.makeText(MainActivity.this, "Failed to download " + file.getName(), Toast.LENGTH_LONG).show();
                                            if(file.length() == 0) {
                                                file.delete();
                                            }
                                        }
                                    } catch(Exception e) {
                                        Log.d("Downloading Failed", file.getName());
                                        Toast.makeText(MainActivity.this, "Failed to download " + file.getName(), Toast.LENGTH_LONG).show();
                                        if(file.length() == 0) {
                                            file.delete();
                                        }
                                        e.printStackTrace();
                                    }
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            } else {
                                //files already existed no need to fetch
                                runBools[2] = true;
                                if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                    btnRunModels.setEnabled(true);
                                }
                            }
                        } else {
                            //no attack data being used
                            truthValue[0] = true;
                            runBools[2] = true;
                            if(runBools[0] && runBools[1] && runBools[2]) { //turn on run model button once all other are ready
                                btnRunModels.setEnabled(true);
                            }
                        }
                    }
                });
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {

            }
        });

        btnRunModels.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {


                final int[] modelCount = {0};

                for(int i = 0; i < modelNames.length; i++) { //fetches all files in modelNames
                    String fileName = modelNames[i] + ".pkl";
                    String url = "http://10.0.2.2:5000/api/fetch-model/"; //host lookback interface
                    if(i < 4) { //number of trained models
                        url += featureTofeat(feature[0]) + "/" + modelNames[i];
                        fileName = featureTofeat(feature[0]) + "_" + fileName;
                    } else { //supporting file
                        url += "none" + "/" + modelNames[i];
                    }

                    File file = new File(appDir, fileName);
                    if(!file.exists()  || file.length() == 0) { //fetch file if doesnt exist or if its empty
                        try {
                            if(!file.exists()) { //create if does not exist
                                file.createNewFile();
                            }
                            DownloadTask download = new DownloadTask();
                            download.execute(fileName, url);
                            try {
                                if(download.get()) { //model downloaded successfully
                                    modelCount[0]++; //loaded model count

                                    //all files loaded safe to run now
                                    if(modelCount[0] == modelNames.length) {
                                        if(userBounds[1] != -1) { //run mult user else run single user
                                            //calls python function in background
                                            runMultiple run = new runMultiple(MainActivity.this);
                                            run.execute(String.valueOf(userBounds[0]), String.valueOf(userBounds[1]), featureTofeat(feature[0]));
                                        } else {
                                            //calls python function in background
                                            runSingle run = new runSingle(MainActivity.this);
                                            run.execute(String.valueOf(userBounds[0]), String.valueOf(attack[0]), featureTofeat(feature[0]), String.valueOf(truthValue[0]));
                                        }
                                    }
                                } else {
                                    Log.d("Downloading Failed", file.getName());
                                    Toast.makeText(MainActivity.this, "Failed to download " + file.getName(), Toast.LENGTH_LONG).show();
                                    if(file.length() == 0) {
                                        file.delete();
                                    }
                                }
                            } catch(Exception e) {
                                Log.d("Downloading Failed", file.getName());
                                Toast.makeText(MainActivity.this, "Failed to download " + file.getName(), Toast.LENGTH_LONG).show();
                                if(file.length() == 0) {
                                    file.delete();
                                }
                                e.printStackTrace();
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    } else { //model already on phone
                        modelCount[0]++; //loaded model count

                        //all files loaded safe to run now
                        if(modelCount[0] == modelNames.length) {
                            if(userBounds[1] != -1) { //run mult user else run single user
                                //calls python function in background
                                runMultiple run = new runMultiple(MainActivity.this);
                                run.execute(String.valueOf(userBounds[0]), String.valueOf(userBounds[1]), featureTofeat(feature[0]));
                            } else {
                                //calls python function in background
                                runSingle run = new runSingle(MainActivity.this);
                                run.execute(String.valueOf(userBounds[0]), String.valueOf(attack[0]), featureTofeat(feature[0]), String.valueOf(truthValue[0]));
                            }
                        }
                    }
                }
            }
        });
    }

    //converts spinner value to python file name
    public static String featureTofeat(String featureName) {
        switch(featureName) {
            case "Coiflets DWT": {
                return "coif";
            }
            case "Alpha Band FT": {
                return "alpha";
            }
            case "Beta Band FT": {
                return "beta";
            }
            case "Delta Band FT": {
                return "delta";
            }
            case "PSD": {
                return "PD";
            }
            case "PCA": {
                return "PCA";
            }
            default: {
                return "error";
            }
        }
    }

    //0 is a true prediction, 1 is a false prediction
    public static boolean doubleToBoolean(double d) {
        return d < 0.5;
    }

    //predicts by highest count ties are assumed true
    public static boolean predict(double[] modelPredictions) {
        int modelTrueCount = 0, modelFalseCount = 0;
        for (double modelPrediction : modelPredictions) {
            if (doubleToBoolean(modelPrediction)) {
                modelTrueCount++;
            } else {
                modelFalseCount++;
            }
        }
        //handles ties assumes true
        return modelTrueCount >= modelFalseCount;
    }

    public class DownloadTask extends AsyncTask<String, String, Boolean> {

        @Override
        protected Boolean doInBackground(String... strings) {

            boolean success = false;
            File appDir = new File("/sdcard/livenessApp");
            String fileName = strings[0]; //fileName to download to

            try
            {
                InputStream input = null;
                try{

                    URL url = new URL(strings[1]); //Server api link to download from
                    HttpURLConnection urlConnection = (HttpURLConnection) url.openConnection();
                    urlConnection.setRequestMethod("POST");
                    urlConnection.setReadTimeout(95 * 1000);
                    urlConnection.setConnectTimeout(95 * 1000);
                    urlConnection.setDoInput(true);
                    urlConnection.setRequestProperty("Accept", "application/octet-stream");
                    urlConnection.setRequestProperty("X-Environment", "android");

                    urlConnection.connect();
                    input = urlConnection.getInputStream();
                    OutputStream output = new FileOutputStream(new File(appDir, fileName));

                    try {
                        byte[] buffer = new byte[1024];
                        int bytesRead = 0;
                        while ((bytesRead = input.read(buffer, 0, buffer.length)) >= 0)
                        {
                            output.write(buffer, 0, bytesRead);

                        }
                        output.close();
                        success = true;
                        Log.d("Downloading Success", fileName);
                    }
                    catch (Exception exception)
                    {
                        Log.e("Error", String.valueOf(exception));
                        output.close();
                    }


                }
                catch (Exception exception)
                {
                    //Toast.makeText(getApplicationContext(), "input exception in catch....."+ exception + "", Toast.LENGTH_LONG).show();
                    Log.e("Error", String.valueOf(exception));

                }
                finally
                {
                    input.close();
                }
            }
            catch (Exception exception)
            {
                Log.e("Error", String.valueOf(exception));
            }

            return success;
        }
    }

}