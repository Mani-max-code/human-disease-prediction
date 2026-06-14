package com.healthpredict;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;

public class ModelTrainer {

    public static void main(String[] args) throws Exception {

        System.out.println("Loading dataset...");

        // Load CSV - skip header line
        RecordReader recordReader = new CSVRecordReader(1);
        File csvFile = new ClassPathResource("data/health_data.csv").getFile();
        recordReader.initialize(new FileSplit(csvFile));

        // 7 input features, label index starts at 7, 3 output classes
        // We predict diabetes (column 7) as main label for simplicity
        int batchSize = 20;
        int labelIndex = 7;
        int numClasses = 2;

        DataSetIterator iterator = new RecordReaderDataSetIterator(
            recordReader, batchSize, labelIndex, numClasses);

        // Normalize data between 0 and 1
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        System.out.println("Building neural network...");

        // Build the neural network
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(7)        // 7 input parameters
                .nOut(16)      // 16 neurons
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder()
                .nIn(16)
                .nOut(8)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder()
                .nIn(8)
                .nOut(2)       // 2 outputs: no risk / risk
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("Training model...");

        // Train for 200 epochs
        iterator.reset();
        for (int epoch = 0; epoch < 200; epoch++) {
            iterator.reset();
            model.fit(iterator);
            if (epoch % 50 == 0) {
                System.out.println("Epoch " + epoch + " complete");
            }
        }

        // Save the model
        File modelFile = new File("src/main/resources/models/health_model.zip");
        modelFile.getParentFile().mkdirs();
        ModelSerializer.writeModel(model, modelFile, true);

        // Save normalizer
       // File normFile = new File("src/main/resources/models/normalizer.bin");


        System.out.println("✅ Model trained and saved successfully!");
    }
}