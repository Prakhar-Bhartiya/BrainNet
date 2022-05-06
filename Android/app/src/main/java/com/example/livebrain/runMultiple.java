package com.example.livebrain;

import android.app.Activity;
import android.os.AsyncTask;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

public class runMultiple extends AsyncTask<String, String, String> {

    PyObject brainModule = null;
    Activity context;

    public runMultiple(Activity context) {
        this.context = context;
    }


    @Override
    protected void onPreExecute() {
        super.onPreExecute();

        Python py = Python.getInstance();
        brainModule = py.getModule("android");
    }

    @Override
    protected String doInBackground(String... strings) {
        String output = brainModule.callAttr("getMultandRun", Integer.parseInt(strings[0]), Integer.parseInt(strings[1]), strings[2]).toJava(String.class);
        return output;
    }

    @Override
    protected void onPostExecute(String output) {
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
        model1Out.setText(output);

        //removes unneeded labels
        model2Out.setText("");
        model3Out.setText("");
        model4Out.setText("");
        truthOut.setText("");
        predictionOut.setText("");
    }
}
