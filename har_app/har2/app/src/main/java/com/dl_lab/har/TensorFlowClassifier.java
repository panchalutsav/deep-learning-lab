package com.dl_lab.har;

import android.content.Context;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TensorFlowClassifier {

    private Interpreter interpreter;
    private static final String MODEL_FILE = "model_forearm.tflite";
    private static final int[] INPUT_SIZE = {1,250,6};
    private static final int OUTPUT_SIZE = 8;

    public TensorFlowClassifier(final Context context) {
        try {
            interpreter = new Interpreter(loadModelFile(context.getAssets(), MODEL_FILE));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        try (InputStream inputStream = assetManager.open(modelPath)) {
            int modelSize = inputStream.available();
            ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelSize);
            byte[] buffer = new byte[modelSize];
            inputStream.read(buffer);
            modelBuffer.put(buffer);
            return modelBuffer.order(ByteOrder.nativeOrder());
        }
    }

    public float[] predictProbabilities(float[] data) {
        float[][] result = new float[1][OUTPUT_SIZE];

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE[0] * INPUT_SIZE[1] * INPUT_SIZE[2] * Float.BYTES)
                .order(ByteOrder.nativeOrder());

        inputBuffer.asFloatBuffer().put(data);

        interpreter.run(inputBuffer, result);

        return result[0];

    }
}
