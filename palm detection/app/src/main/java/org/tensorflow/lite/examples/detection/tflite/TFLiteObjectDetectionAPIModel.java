/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Vector;
import java.util.stream.DoubleStream;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.R;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static float[][] candidate_anchors = new float[2944][4];
  private static final Logger LOGGER = new Logger();

  // Only return this many results.
  private static final int NUM_DETECTIONS = 1;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][][] outputReg = new float[1][2944][18];
  private float[][][] outputClf = new float[1][2944][1];

  private ByteBuffer imgData;

  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];

    try (Scanner scanner = new Scanner(assetManager.open("anchors.csv"));) {
    int x = 0;
      while (scanner.hasNextLine()) {
//        records.add(getRecordFromLine());
        String[] cols = scanner.nextLine().split(",");
        candidate_anchors[x++] = new float[]{Float.valueOf(cols[0]), Float.valueOf(cols[1]), Float.valueOf(cols[2]), Float.valueOf(cols[3])};
      }
    }

    return d;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputReg);
    outputMap.put(1, outputClf);
//    outputMap.put(0, outputLocations);
//    outputMap.put(1, outputClasses);
//    outputMap.put(2, outputScores);
//    outputMap.put(3, numDetections);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
//    ArrayList<float[]> candidate_detect = new ArrayList<float[]>(2994);
    float[][] candidate_detect = new float[2944][18];

//    LOGGER.d("");

//    int max_idx = 0;
    for (int i = 0; i < outputClf[0].length; i++) {
      double x = 1 / (1 + Math.exp(-outputClf[0][i][0]));

      if (x > 0.9) {
//        LOGGER.d("x = "+x);
        candidate_detect[i] = outputReg[0][i];
      }
    }

    //if (candidate_detect.size() == 0)
    //  return recognitions;

    int max_idx = 0;
    float max_suggestion = candidate_detect[0][3];
    for (int i = 0; i < candidate_detect.length; i++) {
      if (candidate_detect[i][3] > max_suggestion) {
        max_idx = i;
      }
    }

    if (max_idx == 0)
      return recognitions;

    float dx, dy, w, h, side, center_wo_offst_x, center_wo_offst_y, center_x, center_y;
    dx = candidate_detect[max_idx][0];
    dy = candidate_detect[max_idx][1];
    w = candidate_detect[max_idx][2] * 1.5f;
    h = candidate_detect[max_idx][3] * 1.5f;

    center_wo_offst_x = candidate_anchors[max_idx][0];
    center_wo_offst_y = candidate_anchors[max_idx][1];

    LOGGER.d("dx = "+dx+", dy = "+dy + ", w = "+w+", h = "+h + " center_offset_x = "+ center_wo_offst_x + " center_offset_y = "+ center_wo_offst_y);

    side = Math.max(w, h);

    center_x = dx + side;
    center_y = dy + side;

    recognitions.add(new Recognition("" + max_idx, "palm",  (float)(1 / (1 + Math.exp(-outputClf[0][max_idx][0]))), new RectF(
            center_wo_offst_x*inputSize - side/2,
            center_wo_offst_y*inputSize - side/2,
            center_wo_offst_x*inputSize+ side/2,
            center_wo_offst_y*inputSize+side/2)));

//    ArrayList<Integer> max_idx = new ArrayList<Integer>();
//    candidate_detect.stream().max(new Comparator<float[]>() {
//      @Override
//      public int compare(float[] o1, float[] o2) {
//        return (int)(o2[3] - o1[3]);
//      }
//    }).ifPresent(ix -> {
//
//      max_idx.add(ix);
//    });
//    outputReg[0]

//    for (int i = 0; i < NUM_DETECTIONS; ++i) {
//      final RectF detection =
//          new RectF(
//              outputLocations[0][i][1] * inputSize,
//              outputLocations[0][i][0] * inputSize,
//              outputLocations[0][i][3] * inputSize,
//              outputLocations[0][i][2] * inputSize);
//      // SSD Mobilenet V1 Model assumes class 0 is background class
//      // in label file and class labels start from 1 to number_of_classes+1,
//      // while outputClasses correspond to class index from 0 to number_of_classes
//      int labelOffset = 1;
//      // added for hand detection
//      final int classLabel = (int) outputClasses[0][i] + labelOffset;
//      if (inRange(classLabel, labels.size(), 0) && inRange(outputScores[0][i], 1, 0)) {
//        recognitions.add(new Recognition("" + i, labels.get(classLabel), outputScores[0][i], detection));
//      }
////      recognitions.add(
////          new Recognition(
////              "" + i,
////              labels.get((int) outputClasses[0][i] + labelOffset),
////              outputScores[0][i],
////              detection));
//    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }

  // added for hand detection
  private boolean inRange(float number, float max, float min) {
    return number < max && number >= min;
  }
  private void setAnchors(){

  }
}
