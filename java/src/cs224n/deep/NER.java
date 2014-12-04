package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev ../data/test");
            return;
        }

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> devData = FeatureFactory.readTestData(args[1]);
        List<Datum> testData = FeatureFactory.readTestData(args[2]);

        // initialize model
        System.out.println("-- Initialized --");
        Map<String, Integer> wordToNum = FeatureFactory.initializeVocab("data/vocab.txt");
        List<String> labels = Arrays.asList("O", "ORG", "PER", "LOC", "MISC");

        int windowSize = 7;     // size of window
        int wordSize = 50;      // size of word vector
        int hiddenSize = 100;   // number of hidden neurons
        int maxEpochs = 50;     // maximum epochs
        double lr0 = 0.001;     // base learning rate
        double lrU0 = 0.001;	// base learning rate for U
        double lrW0 = 0.001;	// base learning rate for W
        double lrL0 = 0.001;	// base learning rate for L
        double tau = .2;        // learning rate decrease speed
        double lambda = 1e-3;   // regularization weight (use 0 for disabled)
        double dropoutX = 0.9;  // probability of keeping X activated during training
        double dropoutV = 0.5;  // probability of keeping V activated during training
        /*WindowModel model = new WindowModel(
                windowSize, wordSize, hiddenSize,
                maxEpochs, lr0, tau, lambda, dropoutX, dropoutV,
                wordToNum, labels);*/
        WindowModel model = new WindowModel(
                windowSize, wordSize, hiddenSize,
                maxEpochs, lrU0, lrW0, lrL0, tau, lambda, dropoutX, dropoutV,
                wordToNum, labels);
        // Standard loading
        SimpleMatrix allVecs = FeatureFactory.readWordVectors("data/wordVectors.txt");
        model.loadVocab(allVecs);
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
        model.train(trainData, devData);

        model.dumpVocab("data/saved-vocab.csv");
        model.dumpWeightsU("data/saved-U.csv");
        model.dumpWeigthsW("data/saved-W.csv");

        System.out.println("-- Test data --");
        model.test(testData, "test_prediction.out");
    }
}
