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

        //	read the train and test data
        //TODO: Implement this function (just reads in vocab and word vectors)

        // initialize model
        System.out.println("-- Initialized --");
        Map<String, Integer> wordToNum = FeatureFactory.initializeVocab("data/vocab.txt");
        int windowSize = 7;
        int wordSize = 50;
        int hiddenSize = 100;
        int maxEpochs = 10;
        double lr0 = 0.01; // base learning rate
        double tau = 1; // parameter for learning rate decrease speed
        WindowModel model = new WindowModel(
                windowSize, wordSize, hiddenSize,
                maxEpochs, lr0, tau,
                wordToNum, Arrays.asList("O", "ORG", "PER", "LOC", "MISC"));

        // Standard loading
        //SimpleMatrix allVecs = FeatureFactory.readWordVectors("data/wordVectors.txt");
        //model.loadVocab(allVecs);
        //model.initWeights();

        // Loading from files
        model.loadVocab(SimpleMatrix.loadCSV("data/saved-vocab.csv"));
        model.loadWeightsU(SimpleMatrix.loadCSV("data/saved-U.csv"));
        model.loadWeightsW(SimpleMatrix.loadCSV("data/saved-W.csv"));

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
