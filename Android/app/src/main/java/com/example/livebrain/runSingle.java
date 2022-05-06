package com.example.livebrain;

import android.app.Activity;
import android.os.AsyncTask;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

public class runSingle extends AsyncTask<String, String, double[]> {

    PyObject brainModule = null;
    Activity context;
    String truthValue = "";

    public runSingle(Activity context) {
        this.context = context;
    }


    @Override
    protected void onPreExecute() {
        super.onPreExecute();

        Python py = Python.getInstance();
        brainModule = py.getModule("android");
    }

    @Override
    protected double[] doInBackground(String... strings) {
        truthValue = strings[3];
        double[] output = brainModule.callAttr("getSubandRun", Integer.parseInt(strings[0]), Integer.parseInt(strings[1]), strings[2]).toJava(double[].class);
        return output;
    }

    @Override
    protected void onPostExecute(double[] output) {
        super.onPostExecute(output);

        //Inputs
        Button btnRunModels = context.findViewById(R.id.btnRunModels);
        Spinner featuresSpinner = context.findViewById(R.id.spinnerFeatures);
        Spinner attackSpinner = context.findViewById(R.id.spinnerAttack);
        EditText editTextUserSelect = context.findViewById(R.id.editTextUsers);

        //Outputs
        TextView model1Out = context.findViewById(R.id.textViewModel1);
        TextView model2Out = context.findViewById(R.id.textViewModel2);
        TextView model3Out = context.findViewById(R.id.textViewModel3);
        TextView model4Out = context.findViewById(R.id.textViewModel4);
        TextView predictionOut = context.findViewById(R.id.textViewDecision);
        TextView truthOut = context.findViewById(R.id.textViewTruth);

        //UI control
        btnRunModels.setEnabled(false);
        featuresSpinner.setEnabled(true);
        attackSpinner.setEnabled(true);
        editTextUserSelect.setEnabled(true);

        //UI output
        model1Out.setText("LogReg: " + MainActivity.doubleToBoolean(output[0]));
        model2Out.setText("KMeans: " + MainActivity.doubleToBoolean(output[1]));
        model3Out.setText("SVM: " + MainActivity.doubleToBoolean(output[2]));
        model4Out.setText("KNN: " + MainActivity.doubleToBoolean(output[3]));
        truthOut.setText("Truth: " + truthValue);
        predictionOut.setText("Verdict: " + MainActivity.predict(output));
    }
}
