package com.dl_lab.har;

import android.annotation.SuppressLint;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.nfc.Tag;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.widget.TextView;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity implements SensorEventListener, TextToSpeech.OnInitListener {

    private static final int N_SAMPLES = 250;
    private static final String TAG = "MainActivity";
    private static List<Float> acc_x;
    private static List<Float> acc_y;
    private static List<Float> acc_z;
    private static List<Float> gyr_x;
    private static List<Float> gyr_y;
    private static List<Float> gyr_z;
    private static List<Float> lx, ly, lz;
    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGyroscope, mLinearAcceleration;
    private static final int SENSOR_DELAY_MICROS = 20000;

    private TextView climbingdown_tv,climbingup_tv, jumping_tv,lying_tv , standing_tv ,sitting_tv, running_tv, walking_tv;
    private TextView probabilitiesTextView;

    private TextToSpeech textToSpeech;
    private float[] result;
    private TensorFlowClassifier classifier;

    private final String[] labels = {"climbing down", "climbing up", "jumping", "lying","standing" ,"sitting", "running", "walking"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initLayoutItems();

        acc_x = new ArrayList<>();
        acc_y = new ArrayList<>();
        acc_z = new ArrayList<>();
        gyr_x = new ArrayList<>();
        gyr_y = new ArrayList<>();
        gyr_z = new ArrayList<>();

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);




        classifier = new TensorFlowClassifier(getApplicationContext());

        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);

        textToSpeech = new TextToSpeech(this, this);
        textToSpeech.setLanguage(Locale.US);
    }

    private void initLayoutItems() {
        climbingdown_tv = findViewById(R.id.climbingdown_TextView);
        climbingup_tv = findViewById(R.id.climbingup_TextView);
        jumping_tv = findViewById(R.id.jumping_TextView);
        lying_tv  = findViewById(R.id.lying_TextView);
        standing_tv = findViewById(R.id.standing_TextView);
        sitting_tv = findViewById(R.id.sitting_TextView);
        running_tv = findViewById(R.id.running_TextView);
        walking_tv = findViewById(R.id.walking_TextView);
    }

    @Override
    public void onInit(int status) {
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (result == null || result.length == 0) {
                    return;
                }
                float max = -1;
                int idx = -1;
                for (int i = 0; i < result.length; i++) {
                    if (result[i] > max) {
                        idx = i;
                        max = result[i];
                    }
                }
                textToSpeech.speak(labels[idx], TextToSpeech.QUEUE_ADD, null, Integer.toString(new Random().nextInt()));
//                activityTextView.setText(labels[idx]);
            }
        }, 2000, 5000);
    }

    protected void onPause() {
        getSensorManager().unregisterListener(this);
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);

    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        mSensorManager.unregisterListener(this);
    }



    @Override
    public void onSensorChanged(SensorEvent event) {

        Sensor sensor = event.sensor;
        if(sensor.getType() == Sensor.TYPE_ACCELEROMETER){
            acc_x.add(event.values[0]);
            acc_y.add(event.values[1]);
            acc_z.add(event.values[2]);
        }
        else if (sensor.getType() == Sensor.TYPE_GYROSCOPE){
            gyr_x.add(event.values[0]);
            gyr_y.add(event.values[1]);
            gyr_z.add(event.values[2]);
        }

        activityPrediction(); // call for prediction when there is a change in the data else let it rest
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    @SuppressLint("SetTextI18n")
    private void activityPrediction() {
        if (acc_x.size() >= N_SAMPLES && acc_y.size() >= N_SAMPLES && acc_z.size() >= N_SAMPLES &&
                gyr_x.size() >= N_SAMPLES && gyr_y.size() >= N_SAMPLES && gyr_z.size() >= N_SAMPLES) {
            List<Float> data = new ArrayList<>();

            data.addAll(acc_x.subList(0, N_SAMPLES));
            data.addAll(acc_y.subList(0, N_SAMPLES));
            data.addAll(acc_z.subList(0, N_SAMPLES));

            data.addAll(gyr_x.subList(0, N_SAMPLES));
            data.addAll(gyr_y.subList(0, N_SAMPLES));
            data.addAll(gyr_z.subList(0, N_SAMPLES));

//            zScoreNormalize(acc_x);
//            zScoreNormalize(acc_y);
//            zScoreNormalize(acc_z);
//            zScoreNormalize(gyr_x);
//            zScoreNormalize(gyr_y);
//            zScoreNormalize(gyr_z);

            result = classifier.predictProbabilities(toFloatArray(data));
            Log.i(TAG, "predict activity:"+ Arrays.toString(result));
            final StringBuilder resultString = new StringBuilder("Result Array: ");
            for (float value : result) {
                resultString.append(value).append(",    ");
            }
            climbingdown_tv.setText("Climbing Down: \t" + round(result[0],2));
            climbingup_tv.setText("Climbing Up: \t" + round(result[1],2));
            jumping_tv.setText("Jumping: \t" + round(result[2],2));
            lying_tv.setText("Lying: \t" + round(result[3],2));
            standing_tv.setText("Standing: \t" + round(result[4],2));
            sitting_tv.setText("Sitting: \t" + round(result[5],2));;
            running_tv.setText("Running: \t" + round(result[6],2));
            walking_tv.setText("Walking: \t" + round(result[7],2));

            data.clear();
            acc_x.clear();
            acc_y.clear();
            acc_z.clear();
            gyr_x.clear();
            gyr_y.clear();
            gyr_z.clear();
        }
    }


    private float round(float value, int decimal_places) {
        BigDecimal bigDecimal=new BigDecimal(Float.toString(value));
        bigDecimal = bigDecimal.setScale(decimal_places, BigDecimal.ROUND_HALF_UP);
        return bigDecimal.floatValue();
    }

    private float[] toFloatArray(List<Float> list) {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private SensorManager getSensorManager() {
        return (SensorManager) getSystemService(SENSOR_SERVICE);
    }

    private void zScoreNormalize(List<Float> data){
        float mean = 0;
        for (float value: data){
            mean+=value;
        }
        mean =(float) mean/data.size();

        float sd = 0;
        for (float value: data){
            sd += (float) Math.pow(value-mean, 2);
        }
        sd = (float) Math.sqrt(sd/data.size());

        for (int i=0; i<data.size(); i++){
            data.set(i, (data.get(i) - mean)/ sd);
        }
    }


}
