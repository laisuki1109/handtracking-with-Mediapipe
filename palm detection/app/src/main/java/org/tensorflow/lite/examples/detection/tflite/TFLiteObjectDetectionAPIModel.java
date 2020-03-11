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
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.RectF;
import android.icu.text.SimpleDateFormat;
import android.media.audiofx.EnvironmentalReverb;
import android.os.Environment;
import android.os.Trace;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Vector;


import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static float[][] anchors = new float[2944][4];
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

  private float[][] outputJoints = new float[1][42];
//  private float[][] outputJoints = new float[1][42];

  private ByteBuffer imgData;

  private ByteBuffer imgLandmarkData;

  private Interpreter tfLite;

  private Interpreter landmarkTfLite;

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
      d.landmarkTfLite = new Interpreter(loadModelFile(assetManager, "hand_landmark.tflite"));
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

    d.imgLandmarkData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * 4);
    d.imgLandmarkData.order(ByteOrder.nativeOrder());

    d.tfLite.setNumThreads(NUM_THREADS);
    d.landmarkTfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];

    // read anchors.csv
    try (Scanner scanner = new Scanner(assetManager.open("anchors.csv"));) {
    int x = 0;
      while (scanner.hasNextLine()) {
//        records.add(getRecordFromLine());
        String[] cols = scanner.nextLine().split(",");
        anchors[x++] = new float[]{Float.valueOf(cols[0]) , Float.valueOf(cols[1]) , Float.valueOf(cols[2]), Float.valueOf(cols[3])};
      }
    }
    return d;
  }

  @Override
  public List<Recognition> recognizeImage(Bitmap bitmap, Bitmap oribmp) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.

//    try {
//      LOGGER.d("path = "+Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      File f = new File(Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      f.createNewFile();
//      FileOutputStream fos = new FileOutputStream(f);
//      if (bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)){
//        fos.flush();
//      }
//      fos.close();
//    } catch (FileNotFoundException e) {
//      e.printStackTrace();
//    } catch (IOException e) {
//      e.printStackTrace();
//    }

    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // put the image in imgData
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
//          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

          imgData.putFloat(((((pixelValue >> 16) & 0xFF) / 255f) - .5f) * 2);
          imgData.putFloat(((((pixelValue >> 8) & 0xFF) / 255f) - .5f) * 2);
          imgData.putFloat((((pixelValue & 0xFF) / 255f) - .5f) * 2);

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
    LOGGER.d("tfLite.getOutputTensorCount() = "+tfLite.getOutputTensorCount());
    LOGGER.d("getOutputTensor(0) = "+tfLite.getOutputTensor(0).name());
    LOGGER.d("getOutputTensor(1) = "+tfLite.getOutputTensor(1).name());
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
//    ArrayList<float[]> candidate_detect = new ArrayList<float[]>(2994);
    ArrayList<float[]> candidate_detect_array = new ArrayList<float[]>();
    ArrayList<float[]> candidate_anchors_array = new ArrayList<float[]>();
//    float[][] candidate_detect = new float[2944][18];

    float[] clf = new float[outputClf[0].length];

    int max_idx = 0;
    double max_suggestion = 0;
    int count = 0;
    int clf_max_idx = 0;

  // finding the best result of detecting hand
    for (int i = 0; i < outputClf[0].length; i++) {
      clf[i] = outputClf[0][i][0];

      float x = 1 / Double.valueOf(1 + Math.exp(-outputClf[0][i][0])).floatValue();
//      LOGGER.d("sigm [" + i + "] = "+ x);

      if (x > 0.7f) {
        LOGGER.d("detecion_mask = "+i+" "+outputClf[0][i][0]+" "+x);
        LOGGER.d("detecion_mask = "+outputReg[0][i][0]+" "+outputReg[0][i][1]+" "+outputReg[0][i][2]);
        count++;
        LOGGER.d("x = "+x+", i = "+i);
        candidate_detect_array.add(outputReg[0][i]);
        candidate_anchors_array.add(anchors[i]);
        if (x > max_suggestion) {
          max_suggestion = x;
          max_idx = candidate_detect_array.size()-1;
          clf_max_idx = i ;
          LOGGER.d("clf_max_idx = " + clf_max_idx);
//          LOGGER.d("anchors["+i+"] = "+anchors[i][0]+", "+anchors[i][1]);
        }
      }
    }
    LOGGER.d("count = "+count);
    if (candidate_detect_array.size() == 0)
      return recognitions;

    float[][] candidate_detect = new float[candidate_detect_array.size()][18];
    float[][] candidate_anchors = new float[candidate_anchors_array.size()][4];

    for (int i=0;i<candidate_anchors_array.size();i++) {
      candidate_detect[i] = candidate_detect_array.get(i);
      candidate_anchors[i] = candidate_anchors_array.get(i);
      LOGGER.d("candidate_detect["+i+"][3] = "+candidate_detect[i][3]);
    }

    LOGGER.d("max_idx = "+ max_idx);
    LOGGER.d("max_suggestion = "+ max_suggestion);

  // palm detection result
    float dx, dy, w, h, side, center_wo_offst_x, center_wo_offst_y;
    dx = candidate_detect[max_idx][0];
    dy = candidate_detect[max_idx][1];
    w = candidate_detect[max_idx][2];
    h = candidate_detect[max_idx][3];
    center_wo_offst_x = candidate_anchors[max_idx][0] * inputSize;
    center_wo_offst_y = candidate_anchors[max_idx][1] * inputSize;
    side = Math.max(w, h) * 1.5f;
    int keypoint_side = 5;
    LOGGER.d("dx = "+dx+", dy = "+dy + ", w = "+w+", h = "+h + " center_offset_x = "+ center_wo_offst_x + " center_offset_y = "+ center_wo_offst_y);
    LOGGER.d("max_id = " + max_idx);

    // mark 3 keypoints in palm (kp0, kp1, kp2)
    Point[] kp = new Point[]{
            new Point(candidate_detect[max_idx][4] + center_wo_offst_x, candidate_detect[max_idx][5] + center_wo_offst_y),
            new Point(candidate_detect[max_idx][6] + center_wo_offst_x, candidate_detect[max_idx][7]+ center_wo_offst_y),
            new Point(candidate_detect[max_idx][8] + center_wo_offst_x, candidate_detect[max_idx][9]+ center_wo_offst_y),
    };
    LOGGER.d("kp0 x = " + kp[0].x + " kp0 y = " + kp[0].y );
    LOGGER.d("kp2 x = " + kp[2].x + " kp2 y = " + kp[2].y );

    // calculate the source and triangle for img landmark
    float[][] source_raw = getTriangle((float)kp[0].x, (float)kp[0].y, (float)kp[2].x, (float)kp[2].y, side);
    float scale = Math.max(oribmp.getHeight()/256f , oribmp.getWidth()/256f) ;
    LOGGER.d("scale = " + scale);
    MatOfPoint2f source = new MatOfPoint2f(new Point(source_raw[0][0]* scale, source_raw[0][1]*scale),
                                  new Point(source_raw[1][0]*scale, source_raw[1][1]*scale),
                                  new Point(source_raw[2][0]*scale, source_raw[2][1]*scale));

    LOGGER.d("source = " + source);
    LOGGER.d("source_raw = " + Arrays.toString(source_raw[0]) +  Arrays.toString(source_raw[1])+  Arrays.toString(source_raw[2]));

    MatOfPoint2f targetTriangle = new MatOfPoint2f(new Point(128f, 128f),
            new Point(128f, 0),
            new Point(0, 128f));

    //print 7 initial keypoints for palm
    for (int k = 4;k < candidate_detect[max_idx].length; k+=2 ){
      float kp_x = candidate_detect[max_idx][k]+center_wo_offst_x;
      float kp_y = candidate_detect[max_idx][k+1]+center_wo_offst_y;
      LOGGER.d("keypoint = "+ kp_x + " , " + kp_y );

      recognitions.add(new Recognition("" + k, "keypoint",  (float)(1 / (1 + Math.exp(-outputClf[0][clf_max_idx][0]))), new RectF(
              candidate_detect[max_idx][k]+center_wo_offst_x - keypoint_side,
              candidate_detect[max_idx][k+1]+center_wo_offst_y - keypoint_side,
              candidate_detect[max_idx][k]+center_wo_offst_x + keypoint_side,
              candidate_detect[max_idx][k+1]+center_wo_offst_y + keypoint_side)));
    }
    // print the palm bbox
    recognitions.add(new Recognition("" + 0, "palm",  (float)(1 / (1 + Math.exp(-outputClf[0][clf_max_idx][0]))), new RectF(
            center_wo_offst_x - side/2,
            center_wo_offst_y - side/2,
            center_wo_offst_x + side/2,
            center_wo_offst_y + side/2)));
//
//    try {
//      LOGGER.d("path = "+Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      File f = new File(Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      f.createNewFile();
//      FileOutputStream fos = new FileOutputStream(f);
//      if (bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)){
//        fos.flush();
//      }
//      fos.close();
//    } catch (FileNotFoundException e) {
//      e.printStackTrace();
//    } catch (IOException e) {
//      e.printStackTrace();
//    }



    Mat mtr = Imgproc.getAffineTransform(source, targetTriangle);
    LOGGER.d("mtr = "+ mtr.dump());
    // cal Mtr inverse for returning the point
    Mat mtrr = new Mat();
    mtrr.create(2, 3, CvType.CV_32FC1);
    mtr.convertTo(mtrr, CvType.CV_32FC1);


    // add row {0,0,0}
    // change (2,2) --> 1 && convert to float 32
    Mat row = new Mat(1,3,CvType.CV_32FC1);
    row.put(0, 0, 0f);
    row.put(0, 1, 0f);
    row.put(0, 2, 1f);
    mtrr.push_back(row);

    Mat minv  = mtrr.inv(); //inverse and transpose
    Mat mtri = minv.rowRange(0, 2);
    LOGGER.d("mtr = "+ mtr.dump());
    LOGGER.d("mtrr = "+ mtrr.dump());

//    LOGGER.d("minv = "+ minv.dump());
//    LOGGER.d("mtri = "+mtri.dump());

    Matrix mMtr = new Matrix();
    transformMatrix(mtrr, mMtr);
    float[] range = {1,0,0,0,1,0,0,0,1};
    mMtr.mapPoints(range);
    LOGGER.d("mMtr = " + Arrays.toString(range));
    Matrix mRot = new Matrix();
    mRot.postRotate(90);
    mRot.postTranslate(oribmp.getHeight()+((oribmp.getWidth()-oribmp.getHeight())/2f), 0);

    // make bitmap for original bitmap 1280*720 to 1280*1280 pad image
//    Bitmap origBitmap = BitmapFactory.decodeFile(Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/20200203_204820.jpg");
    Bitmap origBitmap = oribmp;
    Bitmap bitmapPad = Bitmap.createBitmap(1280, 1280, Bitmap.Config.ARGB_8888);
    Bitmap affineBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);

    final Canvas rotatedCanvas = new Canvas(bitmapPad);
    rotatedCanvas.drawBitmap(origBitmap, mRot, null);

    final Canvas applyAffineCanvas = new Canvas(affineBitmap);
    applyAffineCanvas.drawBitmap(bitmapPad, mMtr, null);

//    try {
//      LOGGER.d("path = "+Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      File f = new File(Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      f.createNewFile();
//      FileOutputStream fos = new FileOutputStream(f);
//      if (bmp.compress(Bitmap.CompressFormat.JPEG, 100, fos)){
//        fos.flush();
//      }
//      fos.close();
//    } catch (FileNotFoundException e) {
//      e.printStackTrace();
//    } catch (IOException e) {
//      e.printStackTrace();
//    }

// store imglandmark as byte buffer for tensorflow
    affineBitmap.getPixels(intValues, 0, 256, 0, 0, 256, 256);
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
          imgData.putFloat(((((pixelValue >> 16) & 0xFF) / 255f) - .5f) * 2);
          imgData.putFloat(((((pixelValue >> 8) & 0xFF) / 255f) - .5f) * 2);
          imgData.putFloat((((pixelValue & 0xFF) / 255f) - .5f) * 2);
        }
      }
    }

    Object[] inputArray2 = {imgData};
    Map<Integer, Object> outputJointMap = new HashMap<>();
    outputJointMap.put(0, outputJoints);

    // Run the inference call.
    Trace.beginSection("run");
    landmarkTfLite.runForMultipleInputsOutputs(inputArray2, outputJointMap);
    Trace.endSection();

    LOGGER.d("outputJoints[0] = "+Arrays.toString(outputJoints[0]));
    // print the joints in the landmark bmp
    affineBitmap =affineBitmap.copy(Bitmap.Config.ARGB_8888, true);
    for(int i = 0; i < outputJoints[0].length; i+=2) {
      affineBitmap.setPixel(Float.valueOf(outputJoints[0][i]).intValue(), Float.valueOf(outputJoints[0][i+1]).intValue(), Color.GREEN);
    }

    // save the img landmark in the mobile's DCIM folder
//    try {
//      LOGGER.d("path = "+Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      File f = new File(Environment.getExternalStorageDirectory()+"/"+Environment.DIRECTORY_DCIM+"/"+new SimpleDateFormat("EEEEE dd MMMMM yyyy HH:mm:ss.SSSZ").format(new Date())+".jpg");
//      f.createNewFile();
//      FileOutputStream fos = new FileOutputStream(f);
//      if (affineBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)){
//        fos.flush();
//      }
//      fos.close();
//    } catch (FileNotFoundException e) {
//      e.printStackTrace();
//    } catch (IOException e) {
//      e.printStackTrace();
//    }

    // put the outputJoints in the joints array list
    LOGGER.d("GESTURE = " + regconizeGesture(outputJoints[0]));



    // print the result to the screen
//    for (int i = 0; i < joints.size(); i++){
//      recognitions.add(new Recognition("" + (i), "origPoint", 1f, new RectF(
//        (float) joints.get(i).x - keypoint_side,
//        (float) joints.get(i).y - keypoint_side,
//        (float) joints.get(i).x + keypoint_side,
//        (float) joints.get(i).y + keypoint_side)));
//    }





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

  static void transformMatrix(Mat src, Matrix dst) {

    int columns = src.cols();
    int rows = src.rows();

    float[] values = new float[columns * rows];
    int index = 0;
    for (int x = 0; x < columns; x++)
      for (int y = 0; y < rows; y++) {
        double[] value = src.get(x, y);
        values[index] = (float) value[0];

        index++;
      }

    dst.setValues(values);
  }

  public static float[][] getTriangle(float kp0_x, float kp0_y ,float kp2_x, float kp2_y, float dist) {
    // cal triangle with kp x-coordinates
    float[][] triangle = new float[3][2];
    float x = kp2_x - kp0_x;
    float y = kp2_y - kp0_y;

    LOGGER.d("kp2 - kp0 = " + x + ", " +y);

    //    cal matrix norm
    float [][] dir_v = new float [1][2];
    dir_v[0][0] = x /(float) (Math.sqrt( Math.pow(x, 2) +Math.pow(y, 2)));
    dir_v[0][1] = y /(float) (Math.sqrt( Math.pow(x, 2) +Math.pow(y, 2)));

    LOGGER.d("norm "+(Math.sqrt( Math.pow(x, 2) +Math.pow(y, 2))));

    float [][] rotate = new float [2][2]; //R90.T
    rotate[0][0] = 0f;
    rotate[0][1] = -1f;
    rotate[1][0] = 1f;
    rotate[1][1] = 0f;

    //    cal  dir_v @ rotate
    float[][] dir_v_r = new float[dir_v.length][rotate[0].length];
    for(int i = 0; i < dir_v.length; i++) {
      for (int j = 0; j < rotate[0].length; j++) {
        for (int k = 0; k < dir_v[0].length; k++) {
          dir_v_r[i][j] += dir_v[i][k] * rotate[k][j];
        }
      }
    }
    // b = kp2 + dir_v * dist
    float [][] b = new float [1][2];
    b[0][0] = kp2_x + dir_v[0][0] * dist - ((kp0_x-kp2_x) * .2f);
    b[0][1] = kp2_y + dir_v[0][1] * dist - ((kp0_y-kp2_y) * .2f);

    // c = kp2 + dir_v_r * dist
    float [][] c = new float [1][2];
    c[0][0] = kp2_x + dir_v_r[0][0] * dist - ((kp0_x-kp2_x) * .2f);
    c[0][1] = kp2_y + dir_v_r[0][1] * dist - ((kp0_y-kp2_y) * .2f);

    triangle[0] = new float[]{kp2_x - ((kp0_x-kp2_x) *.2f), kp2_y - ((kp0_y-kp2_y) * .2f) };
    triangle[1] = b[0];
    triangle[2] = c[0];

    return triangle;
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

  public static Mat norm(Mat in) { // only support CV_8UC4
    Mat out = new Mat(in.rows(), in.cols(), CvType.CV_32FC4);
    for(int i=0;i<in.rows();i++) {
      for(int j=0;j<in.cols();j++) {
        byte b[] = new byte[4];
        out.put(i, j, new float[]{
                2 * ((b[0]/255) - 0.5f),
                2 * ((b[1]/255) - 0.5f),
                2 * ((b[2]/255) - 0.5f),
                2 * ((b[3]/255) - 0.5f),
        });
      }

    }
    return out;
  }

  public static Mat norm2(Mat in) { // only support CV_8UC4
    Mat out = new Mat(in.rows(), in.cols(), CvType.CV_32FC4);
    for(int i=0;i<in.rows();i++) {
      for(int j=0;j<in.cols();j++) {
//        byte b[] = new byte[4];
        out.put(i, j, new float[]{
                (((float)in.get(i,j)[0])),
                (((float)in.get(i,j)[1])),
                (((float)in.get(i,j)[2])),
                (((float)in.get(i,j)[3])),
        });
//        LOGGER.d("out = "+ out.get(i,j)[0]);
      }

    }
    return out;
  }

  public static String regconizeGesture(float[] joints) {
    // finger states
    boolean thumbIsOpen = false;
    boolean firstFingerIsOpen = false;
    boolean secondFingerIsOpen = false;
    boolean thirdFingerIsOpen = false;
    boolean fourthFingerIsOpen = false;
    LOGGER.d("gesture joints = "+ Arrays.toString(joints));

    double pseudoFixKeyPoint = joints[2*2]; //compare x
    if (joints[3*2] < pseudoFixKeyPoint && joints[4*2] < pseudoFixKeyPoint)
    {
      thumbIsOpen = true;
    }

    pseudoFixKeyPoint = joints[6*2 + 1]; //compare y
    if (joints[7*2 + 1] < pseudoFixKeyPoint && joints[8*2 + 1] < pseudoFixKeyPoint)
    {
      firstFingerIsOpen = true;
    }

    pseudoFixKeyPoint = joints[10*2 + 1]; //compare y
    if (joints[11*2 + 1] < pseudoFixKeyPoint && joints[12*2 + 1] < pseudoFixKeyPoint)
    {
      secondFingerIsOpen = true;
    }

    pseudoFixKeyPoint = joints[14*2 + 1]; //compare y
    if (joints[15*2 + 1] < pseudoFixKeyPoint && joints[16*2 + 1] < pseudoFixKeyPoint)
    {
      thirdFingerIsOpen = true;
    }

    pseudoFixKeyPoint = joints[18*2 + 1]; //compare y
    if (joints[19*2 + 1] < pseudoFixKeyPoint && joints[20*2 + 1] < pseudoFixKeyPoint)
    {
      fourthFingerIsOpen = true;
    }
    // Hand gesture recognition
    if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen)
    {
      return "FIVE";
    } else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen) {
      return "FOUR";
    }else if (thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
      return "THREE";
    }else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
      return "TWO";
    }else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
      return "ONE";
    }else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
      return "ZERO";
    }else if (thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen) {
      return "LIKE";
    }
    return "no GESTURE detected";
  }
}
