package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
            return;
        }

        // this reads in the train and test datasets
        List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> testData = FeatureFactory.readTestData(args[1]);

        //	read the train and test data
        //TODO: Implement this function (just reads in vocab and word vectors)

        // initialize model
        Map<String, Integer> wordToNum = FeatureFactory.initializeVocab("data/vocab.txt");
        WindowModel model = new WindowModel(3, 50, 100, 0.001,
                wordToNum, Arrays.asList("O", "ORG", "PER", "LOC", "MISC"));

        SimpleMatrix allVecs = FeatureFactory.readWordVectors("data/wordVectors_sized.txt");
        model.loadVocab(allVecs);
        model.initWeights();
        System.out.println(String.format("U gradient check error: %f", model.computeUgradCheck(100, 1e-5)));
        System.out.println(String.format("W gradient check error: %f", model.computeWgradCheck(100, 1e-5)));
        System.out.println(String.format("X gradient check error: %f", model.computeXgradCheck(100, 1e-5)));

        //TODO: Implement those two functions
        model.train(trainData);
        //model.test(testData);
    }
}