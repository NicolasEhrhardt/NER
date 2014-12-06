package cs224n.deep;

import java.util.*;
import java.io.*;

public class NER {
    
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("USAGE: java -cp classes NER config.properties");
            return;
        }

        InputStream inputStream = new FileInputStream(args[0]);
        Properties properties = new Properties();
        properties.load(inputStream);

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(properties.getProperty("trainFile"));
        List<Datum> holdoutData = FeatureFactory.readTestData(properties.getProperty("holdoutFile"));
        List<Datum> testData = FeatureFactory.readTestData(properties.getProperty("testFile"));

        // initialize model
        System.out.println("-- Initialized --");
        Map<String, Integer> wordToNum = FeatureFactory.initializeVocab("data/vocab.txt");
        List<String> labels = Arrays.asList("O", "ORG", "PER", "LOC", "MISC");

        int windowSize = Integer.valueOf(properties.getProperty("windowSize", "7"));     // size of window
        int wordSize = Integer.valueOf(properties.getProperty("wordSize", "50"));      // size of word vector
        int hiddenSize = Integer.valueOf(properties.getProperty("hiddenSize", "100"));   // number of hidden neurons
        int maxEpochs = Integer.valueOf(properties.getProperty("maxEpochs", "50"));     // maximum epochs
        double lrU0 = Double.valueOf(properties.getProperty("lrU0", "1e-2"));	// base learning rate for U
        double lrW0 = Double.valueOf(properties.getProperty("lrW0", "1e-2"));	// base learning rate for W
        double lrL0 = Double.valueOf(properties.getProperty("lrL0", "1e-2"));	// base learning rate for L
        double tau = Double.valueOf(properties.getProperty("tau", "0.5"));        // learning rate decrease speed
        double lambda = Double.valueOf(properties.getProperty("lambda", "13-3"));   // regularization weight (use 0 for disabled)
        double dropoutX = Double.valueOf(properties.getProperty("dropoutX", "1."));  // probability of keeping X activated during training
        double dropoutZ = Double.valueOf(properties.getProperty("dropoutZ", "1."));  // probability of keeping Z activated during training
        WindowModel model = new WindowModel(
                windowSize, wordSize, hiddenSize,
                maxEpochs, lrU0, lrW0, lrL0,
                tau, lambda, dropoutX, dropoutZ,
                wordToNum, labels);
        // Standard loading
        //model.loadVocab(FeatureFactory.readWordVectors("data/wordVectors.txt"));
        model.initVocab();
        model.initWeights();

        // Loading from files
        //model.loadVocab(SimpleMatrix.loadCSV("data/saved-vocab.csv"));
        //model.loadWeightsU(SimpleMatrix.loadCSV("data/saved-U.csv"));
        //model.loadWeightsW(SimpleMatrix.loadCSV("data/saved-W.csv"));

        //System.out.println("-- Computing gradient checks --");
        //System.out.println(String.format("U gradient check error: %f", model.computeUgradCheck(100, 1e-4)));
        //System.out.println(String.format("W gradient check error: %f", model.computeWgradCheck(1, 1e-4)));
        //System.out.println(String.format("X gradient check error: %f", model.computeXgradCheck(1, 1e-4)));
        
        System.out.println("-- Training data --");
        model.train(trainData, holdoutData);

        model.dumpWeightsU("data/saved-U.csv");
        model.dumpWeigthsW("data/saved-W.csv");
        model.dumpVocab("data/saved-vocab.csv");

        System.out.println("-- Test data --");
        model.test(testData, "test_prediction.out");
    }
}
