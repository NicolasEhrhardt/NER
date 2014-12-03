package cs224n.deep;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.simple.*;

import static cs224n.deep.Utils.*;

public class WindowModel {

    protected SimpleMatrix L, W, U;
    public int windowSize, wordSize, hiddenSize, numWords, K;
    public double lr0, tau; // Base learning rate + time constant
    public List<String> labels;

    public Map<String, Integer> wordToNum;

    public WindowModel(int windowSize, int wordSize, int hiddenSize, double lr0, double tau,
                       Map<String, Integer> wordToNum, List<String> labels) {
        assert (windowSize % 2 == 1);
        this.windowSize = windowSize;
        this.wordSize = wordSize;
        this.hiddenSize = hiddenSize;
        this.lr0 = lr0;
        this.tau = tau;
        this.wordToNum = wordToNum;
        this.numWords = wordToNum.size();
        this.labels = labels;
        this.K = labels.size();
    }

    /**
     * Loaders and dumpers
     */

    public void loadVocab(SimpleMatrix allVec) {
        this.L = allVec;
    }

    public void loadWeightsW(SimpleMatrix W) {
        this.W = W;
    }

    public void loadWeightsU(SimpleMatrix U) {
        this.U = U;
    }

    public void dumpVocab(String filename) throws IOException {
        L.saveToFileCSV(filename);
    }

    public void dumpWeigthsW(String filename) throws IOException {
        W.saveToFileCSV(filename);
    }

    public void dumpWeightsU(String filename) throws IOException {
        U.saveToFileCSV(filename);
    }

    /**
     * Initializes the weights randomly according to the fan-in fan-out rule
     */
    public void initWeights() {
        // Init seed
        Random rand = new Random();
        W = helperInitWeights(windowSize * wordSize + 1, hiddenSize, rand);
        U = helperInitWeights(hiddenSize + 1, K, rand);
    }

    private SimpleMatrix helperInitWeights(int fanin, int fanout, Random rand) {
        double eps = Math.sqrt(6 / (double) (fanin + fanout));
        return SimpleMatrix.random(fanout, fanin, -eps, +eps, rand);
    }

    /**
     * Computation helpers
     */

    // Compute Error

    private static SimpleMatrix computeError(SimpleMatrix y, SimpleMatrix p) {
        return y.minus(p);
    }

    // Compute delta

    private static SimpleMatrix computeDelta(SimpleMatrix errorVector, SimpleMatrix Utruncated, SimpleMatrix z) {
        return elementwiseApplyTanhDerivative(z).elementMult(Utruncated.transpose().mult(errorVector));
    }

    // Compute P

    private SimpleMatrix computePFromX(SimpleMatrix xbiased) {
        return computePFromUWx(U, W, xbiased);
    }

    private static SimpleMatrix computePFromUh(SimpleMatrix U, SimpleMatrix hbiased) {
        return softmax(U.mult(hbiased));
    }

    private static SimpleMatrix computePFromUWx(SimpleMatrix U, SimpleMatrix W, SimpleMatrix xbiased) {
        SimpleMatrix h = elementwiseApplyTanh(W.mult(xbiased));
        return computePFromUh(U, concatenateWithBias(h));
    }

    // Cost

    private static double computeCostFromP(SimpleMatrix y, SimpleMatrix p) {
        return y.elementMult(elementwiseApplyLog(p)).elementSum();
    }

    private static double computeCostFromUh(SimpleMatrix y, SimpleMatrix U, SimpleMatrix hbiased) {
        return computeCostFromP(y, softmax(U.mult(hbiased)));
    }

    private static double computeCostFromUWx(SimpleMatrix y, SimpleMatrix U, SimpleMatrix W, SimpleMatrix xbiased) {
        return computeCostFromUh(y, U, concatenateWithBias(elementwiseApplyTanh(W.mult(xbiased))));
    }

    /**
     * Gradient checks
     */

    // U grad

    private static SimpleMatrix computeUgradFromError(SimpleMatrix errorVector, SimpleMatrix hbiased) {
        return errorVector.mult(hbiased.transpose());
    }

    private static SimpleMatrix computeUgradFromUh(SimpleMatrix y, SimpleMatrix U, SimpleMatrix hbiased) {
        SimpleMatrix p = computePFromUh(U, hbiased);
        SimpleMatrix error = computeError(y, p);
        return computeUgradFromError(error, hbiased);
    }

    public double computeUgradCheck(int trials, double precision) {
        Random rand = new Random();

        List<Double> differences = new ArrayList<Double>();
        for (int i = 0; i < trials; i++) {
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix hbiased = concatenateWithBias(SimpleMatrix.random(hiddenSize, 1, 0, 1, rand));
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);

            SimpleMatrix trueGrad = computeUgradFromUh(y, Utest, hbiased);

            for (int row = 0; row < Utest.numRows(); row++) {
                for (int col = 0; col < Utest.numCols(); col++) {
                    SimpleMatrix eps = new SimpleMatrix(Utest.numRows(), Utest.numCols());
                    eps.zero();
                    eps.set(row, col, precision);

                    SimpleMatrix Upluseps = Utest.plus(eps);
                    SimpleMatrix Uminuseps = Utest.minus(eps);

                    double diff = (computeCostFromUh(y, Upluseps, hbiased) - computeCostFromUh(y, Uminuseps, hbiased))
                            / (2 * precision);

                    differences.add(Math.abs( diff - trueGrad.get(row, col) ));
                }
            }
        }
        return Collections.max(differences);
    }

    // W grad

    private static SimpleMatrix computeWgradFromDelta(SimpleMatrix delta, SimpleMatrix xbiased) {
        return delta.mult(xbiased.transpose());
    }

    private static SimpleMatrix computeWgradFromUWx(SimpleMatrix y, SimpleMatrix U, SimpleMatrix W, SimpleMatrix xbiased) {
        SimpleMatrix Utruncated = withoutLastCol(U);

        SimpleMatrix z = W.mult(xbiased);
        SimpleMatrix p = computePFromUWx(U, W, xbiased);
        SimpleMatrix error = computeError(y, p);
        SimpleMatrix delta = computeDelta(error, Utruncated, z);

        return computeWgradFromDelta(delta, xbiased);
    }

    public double computeWgradCheck(int trials, double precision) {
        Random rand = new Random();

        List<Double> differences = new ArrayList<Double>();
        for (int i = 0; i < trials; i++) {
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix xtest = concatenateWithBias(SimpleMatrix.random(windowSize * wordSize, 1, 0, 1, rand));
            SimpleMatrix Wtest = helperInitWeights(W.numCols(), W.numRows(), rand);
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);

            SimpleMatrix trueGrad = computeWgradFromUWx(y, Utest, Wtest, xtest);

           for (int row = 0; row < Wtest.numRows(); row++) {
                for (int col = 0; col < Wtest.numCols(); col++) {
                    SimpleMatrix eps = new SimpleMatrix(Wtest.numRows(), Wtest.numCols());
                    eps.zero();
                    eps.set(row, col, precision);

                    SimpleMatrix Wpluseps = Wtest.plus(eps);
                    SimpleMatrix Wminuseps = Wtest.minus(eps);

                    double diff = (computeCostFromUWx(y, Utest, Wpluseps, xtest) - computeCostFromUWx(y, Utest, Wminuseps, xtest))
                            / (2 * precision);

                    differences.add(Math.abs( diff - trueGrad.get(row, col) ));
                }
            }
        }
        return Collections.max(differences);
    }

    // X gradient

    private static SimpleMatrix computeXgradFromDelta(SimpleMatrix delta, SimpleMatrix Wtruncated) {
        return Wtruncated.transpose().mult(delta);
    }

    private static SimpleMatrix computeXgradFromUWx(SimpleMatrix y, SimpleMatrix U, SimpleMatrix W, SimpleMatrix xtruncated) {
        SimpleMatrix xbiased = concatenateWithBias(xtruncated);
        SimpleMatrix Utruncated = withoutLastCol(U);
        SimpleMatrix Wtruncated = withoutLastCol(W);

        SimpleMatrix z = W.mult(xbiased);
        SimpleMatrix p = computePFromUWx(U, W, xbiased);
        SimpleMatrix error = computeError(y, p);
        SimpleMatrix delta = computeDelta(error, Utruncated, z);

        return computeXgradFromDelta(delta, Wtruncated);
    }

    public double computeXgradCheck(int trials, double precision) {
        Random rand = new Random();

        List<Double> differences = new ArrayList<Double>();
        for (int i = 0; i < trials; i++) {
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix xtest = SimpleMatrix.random(windowSize * wordSize, 1, 0, 1, rand);
            SimpleMatrix Wtest = helperInitWeights(W.numCols(), W.numRows(), rand);
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);

            SimpleMatrix trueGrad = computeXgradFromUWx(y, Utest, Wtest, xtest);

            for (int row = 0; row < xtest.numRows(); row++) {
                for (int col = 0; col < xtest.numCols(); col++) {
                    SimpleMatrix eps = new SimpleMatrix(xtest.numRows(), xtest.numCols());
                    eps.zero();
                    eps.set(row, col, precision);

                    SimpleMatrix xpluseps = concatenateWithBias(xtest.plus(eps));
                    SimpleMatrix xminuseps = concatenateWithBias(xtest.minus(eps));

                    double diff = (computeCostFromUWx(y, Utest, Wtest, xpluseps) - computeCostFromUWx(y, Utest, Wtest, xminuseps))
                            / (2 * precision);

                    differences.add(Math.abs( diff - trueGrad.get(row, col) ));
                }
            }
        }
        return Collections.max(differences);
    }

    /**
     * Processing helpers
     */

    /**
     * @param data: input list of Datum
     * @return list of examples matching each word to label
     */
    public List<List<Datum>> yieldExamples(List<Datum> data) {
        List<List<Datum>> examples = new ArrayList<List<Datum>>();

        List<Datum> buffer = new ArrayList<Datum>();
        for (Datum datum : data) {

            if (datum.word.equals(FeatureFactory.START_TOKEN)) {
                // Clear buffer if we are restarting a sentence
                buffer.clear();

                // Padding when entering sentence
                for (int i = 0; i < windowSize / 2; i++) {
                    buffer.add(datum);
                }

                // Done for token
                continue;
            }

            if (datum.word.equals(FeatureFactory.END_TOKEN)) {
                // Padding until the end of the sentence
                int paddingNeeded = windowSize - buffer.size();
                for (int i = 0; i < paddingNeeded; i++) {
                    buffer.add(datum);
                }

                // Continue to process until last word is in the middle
                for (int i = 0; i < (windowSize / 2) - paddingNeeded; i++) {
                    examples.add(new ArrayList<Datum>(buffer));
                    buffer.remove(0);
                    buffer.add(datum);
                }

                // Done for token
                continue;
            }

            // Add token if we haven't reached the window size
            if (buffer.size() < windowSize) {
                buffer.add(datum);
            }

            if (buffer.size() == windowSize) {
                // If the buffer is the right size, update the weights and remove the oldest token
                examples.add(new ArrayList<Datum>(buffer));
                buffer.remove(0);
            }
        }

        return examples;
    }

    /**
     * Get word index in the matrix L
     * @param word
     * @return
     */
    private int getWordIndex(String word) {
        if (wordToNum.containsKey(word)) {
            return wordToNum.get(word);
        } else {
            return wordToNum.get(FeatureFactory.UNK_TOKEN);
        }
    }

    /**
     * Get all words index from list of Datum
     * @param buffer
     * @return
     */
    private List<Integer> getLindFromBuffer(List<Datum> buffer) {
        List<Integer> inputIndex = new ArrayList<Integer>();

        for (Datum datum : buffer) {
            // get vector representation
            int index = getWordIndex(datum.word);
            inputIndex.add(index);
        }

        return inputIndex;
    }

    /**
     * Get vector representation (X) of phrase
     * @param inputIndex
     * @return
     */
    private SimpleMatrix getXFromLind(List<Integer> inputIndex) {
        List<SimpleMatrix> inputVectors = new ArrayList<SimpleMatrix>();
        for (int index: inputIndex) {
            SimpleMatrix curVec = L.extractVector(true, index);
            inputVectors.add(curVec.transpose());
        }

        SimpleMatrix x = concatenate(inputVectors);
        return x;
    }

    /**
     * Simplest SGD training
     */

    /**
     * Update U, W, L based on one example, to be used in the SGD
     * @param buffer
     */
    private void updateWeights(List<Datum> buffer, int epoch) {
        List<Integer> inputIndex = getLindFromBuffer(buffer);
        String label = buffer.get(windowSize / 2).label;

        SimpleMatrix x = getXFromLind(inputIndex);
        SimpleMatrix xbiased = concatenateWithBias(x);
        SimpleMatrix z = W.mult(xbiased);

        SimpleMatrix h = elementwiseApplyTanh(z);
        SimpleMatrix hbiased = concatenateWithBias(h);
        SimpleMatrix v = U.mult(hbiased);
        SimpleMatrix p = softmax(v);

        SimpleMatrix Utruncated = withoutLastCol(U);
        SimpleMatrix Wtruncated = withoutLastCol(W);

        assert (labels.contains(label));
        SimpleMatrix y = indicator(K, labels.indexOf(label));
        SimpleMatrix error = computeError(y, p);
        SimpleMatrix delta = computeDelta(error, Utruncated, z);

        SimpleMatrix Ugrad = computeUgradFromError(error, hbiased);
        SimpleMatrix Wgrad = computeWgradFromDelta(delta, xbiased);
        SimpleMatrix Xgrad = computeXgradFromDelta(delta, Wtruncated);
        
        
        // Update U
        double lrU = lr0/(1+((double) epoch/tau));
        U.set(U.plus(lrU, Ugrad));

        // Update W
        double lrW = lr0/(1+((double) epoch/tau));
        W.set(W.plus(lrW, Wgrad));

        // Update x -> L
        double lrL = lr0/(1+((double) epoch/tau));
        x.set(x.plus(lrL, Xgrad));

        for (int i = 0; i < inputIndex.size(); i++) {
            int index = inputIndex.get(i);
            L.insertIntoThis(index, 0, x.extractMatrix(i * wordSize, (i + 1) * wordSize, 0, 1).transpose());
        }
    }

    /**
     * Train the three matrices using the passed training data, stops when the precision decreases on the dev set
     * @param trainData
     * @param devData
     */
    public void train(List<Datum> trainData, List<Datum> devData) {

        SimpleMatrix Usaved = U.copy();
        SimpleMatrix Wsaved = W.copy();
        SimpleMatrix Lsaved = L.copy();

        List<List<Datum>> allExamples = yieldExamples(trainData);
        double precision = 0;
        double new_precision = 0;
        for (int epoch = 0; epoch < 10; epoch++) {
            for (List<Datum> buffer: allExamples) {
                updateWeights(buffer, epoch);
            }
            new_precision = devPrecision(devData);

            System.out.println(String.format("Epochs %d, delta U %f, delta W %f, delta L %f, devSet precision %.2f%%",
                    epoch, U.minus(Usaved).normF(), W.minus(Wsaved).normF(), L.minus(Lsaved).normF(), 100*new_precision));
            if (new_precision < precision){
            	break;
            }
            precision = new_precision;
            Usaved = U.copy();
            Wsaved = W.copy();
            Lsaved = L.copy();
            
        }
    }
    /**
     * Computes the precision (correct guesses / nb of tokens) on the devSet with the current parameters U, W, L
     * @param devData
     * @return
     */
    private double devPrecision(List<Datum> devData) {
        List<List<Datum>> allExamples = yieldExamples(devData);
        double correct_guesses = 0;
        double tokens = 0;
        for (List<Datum> buffer: allExamples) {
            Datum middleWord = buffer.get(windowSize / 2);
            String predictedLabel = predictLabel(buffer);
            if (middleWord.label.equals(predictedLabel)) {
            	correct_guesses++;
            }
            tokens++;
        }
        return correct_guesses/tokens;
    }
    
   /**
    * Prediction and testing
    */

    /**
     * Predict label word in the middle of buffer
     * @param buffer
     * @return
     */
    public String predictLabel(List<Datum> buffer) {
        List<Integer> inputIndex = getLindFromBuffer(buffer);
        SimpleMatrix x = getXFromLind(inputIndex);
        SimpleMatrix xbiased = concatenateWithBias(x);
        SimpleMatrix P = computePFromX(xbiased);
        return labels.get(argmax(P));
    }

    /**
     * Test data and output to file
     * @param testData
     * @param outputFile
     * @throws IOException
     */
    public void test(List<Datum> testData, String outputFile) throws IOException {
    	FileWriter fw = new FileWriter(outputFile);
        List<List<Datum>> allExamples = yieldExamples(testData);

        for (List<Datum> buffer: allExamples) {
            Datum middleWord = buffer.get(windowSize / 2);
            String predictedLabel = predictLabel(buffer);
            fw.write(String.format("%s\t%s\t%s\n", middleWord.word, middleWord.label, predictedLabel));
        }
    	fw.close();
    }
}
