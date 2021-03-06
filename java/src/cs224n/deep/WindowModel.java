package cs224n.deep;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.simple.*;

import static cs224n.deep.Utils.*;

public class WindowModel {

    protected SimpleMatrix L, W, U;
    public int windowSize, wordSize, hiddenSize, maxEpochs, numWords, K;
    public double lr0, tau, lambda; // Base learning rate + time constant
    public double dropoutX, dropoutZ; // Probability of keeping a neuron activated
    public List<String> labels;
    
    public double lrW0, lrL0, lrU0;

    public Map<String, Integer> wordToNum;

    public WindowModel(int windowSize, int wordSize, int hiddenSize,                				// Network parameters
            int maxEpochs, double lrU0, double lrW0, double lrL0, double tau, double lambda,        // Optimization parameters
            double dropoutX, double dropoutZ,
            Map<String, Integer> wordToNum, List<String> labels) {
		assert (windowSize % 2 == 1);
		this.windowSize = windowSize;
		this.wordSize = wordSize;
		this.hiddenSize = hiddenSize;
		this.maxEpochs = maxEpochs;
		this.lrL0 = lrL0;
		this.lrW0 = lrW0;
		this.lrU0 = lrU0;
		this.tau = tau;
		this.lambda = lambda;
		this.dropoutX = dropoutX;
		this.dropoutZ = dropoutZ;
		this.wordToNum = wordToNum;
		this.numWords = wordToNum.size();
		this.labels = labels;
		this.K = labels.size();
		System.out.println(String.format(
                "Window size: %d, word size: %d, hidden size: %d\n" +
                "max epochs: %d, learning rate (U, W, L): %f, %f, %f\n" +
                "tau: %f, lambda: %f, dropout (X, Z) : %.2f, %.2f",
        		windowSize, wordSize, hiddenSize,
                maxEpochs, lrU0, lrW0, lrL0,
                tau, lambda, dropoutX, dropoutZ));
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

    public void initVocab() {
        Random rand = new Random();
        L = helperInitWeights(wordSize, wordToNum.size(), rand);
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
    private void updateWeights(List<Datum> buffer, double lrU, double lrW, double lrL) {
        List<Integer> inputIndex = getLindFromBuffer(buffer);
        String label = buffer.get(windowSize / 2).label;

        SimpleMatrix xorig = getXFromLind(inputIndex);
		SimpleMatrix xkeptind = getDropvector(xorig.numRows(), xorig.numCols(), dropoutX);
		SimpleMatrix x = xorig.elementMult(xkeptind);
        SimpleMatrix ones = new SimpleMatrix(xkeptind);
        ones.set(1.0);
        SimpleMatrix otherx = xorig.elementMult(xkeptind.negative().plus(ones));

        SimpleMatrix xbiased = concatenateWithBias(x);
        SimpleMatrix z = W.mult(xbiased);
        SimpleMatrix zdropped = getDropvector(z.numRows(), z.numCols(), dropoutZ);
        z = z.elementMult(zdropped);

        SimpleMatrix h = elementwiseApplyTanh(z);
        SimpleMatrix hbiased = concatenateWithBias(h);
        SimpleMatrix v = U.mult(hbiased);
        SimpleMatrix p = softmax(v);

        SimpleMatrix Utruncated = withoutLastCol(U);
        SimpleMatrix Wtruncated = withoutLastCol(W);

        SimpleMatrix y = indicator(K, labels.indexOf(label));
        SimpleMatrix error = computeError(y, p);
        SimpleMatrix delta = computeDelta(error, Utruncated, z);

        SimpleMatrix Ugrad = computeUgradFromError(error, hbiased);
        SimpleMatrix Wgrad = computeWgradFromDelta(delta, xbiased);
        SimpleMatrix Xgrad = computeXgradFromDelta(delta, Wtruncated);
        
        
        // Update U
        U.set(U.scale(1. - lambda * lrU).plus(lrU, Ugrad));
        //U = normalizeRows(U, 1);

        // Update W
        W.set(W.scale(1. - lambda * lrW).plus(lrW, Wgrad));
        //W = normalizeRows(W, 1);

        // Update x
        x.set(x.plus(lrL, Xgrad));

        // Keep only gradient for turned on units
        x.set(x.elementMult(xkeptind));

        // Replace x value for turned off units
        x.set(x.plus(otherx));

        // Reinsert x in L
        for (int i = 0; i < inputIndex.size(); i++) {
            int index = inputIndex.get(i);
            L.insertIntoThis(index, 0, x.extractMatrix(i * wordSize, (i + 1) * wordSize, 0, 1).transpose());
        }
    }

    /**
     * Train the three matrices using the passed training data, stops when the precision decreases on the dev set
     * @param trainData
     * @param holdoutData
     */
    public void train(List<Datum> trainData, List<Datum> holdoutData) {

        SimpleMatrix Usaved = U.copy();
        SimpleMatrix Wsaved = W.copy();
        SimpleMatrix Lsaved = L.copy();

        List<List<Datum>> allExamples = yieldExamples(trainData);
        List<List<Datum>> holdoutExamples = yieldExamples(holdoutData);

        double precision = 0;
        double newPrecision = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            long startTime = System.currentTimeMillis();

            // Compute learning rates for this epoch
            double lrU = lrU0 / (1. + ((double) epoch / tau));
            double lrW = lrW0 / (1. + ((double) epoch / tau));
            double lrL = lrL0 / (1. + ((double) epoch / tau));

            int n = 0;
            for (List<Datum> buffer: allExamples) {
                updateWeights(buffer, lrU, lrW, lrL);
                n++;

                if (n % 10000 == 0) {
                    long sofarTime = System.currentTimeMillis();
                    System.out.print(String.format(
                            "\rTraining (%d examples seen in %ds)", n, (sofarTime - startTime) / 1000));
                }
             }

            System.out.print("\rComputing error.");
            newPrecision = getPrecision(holdoutExamples, U.scale(dropoutZ), W.scale(dropoutX));

            long endTime = System.currentTimeMillis();
            System.out.println(String.format(
                    "\rEpoch %d, delta U %f, delta W %f, delta L %f, " +
                            "holdout set precision %.2f%%, (iteration time %ds).",
                    epoch, U.minus(Usaved).normF(), W.minus(Wsaved).normF(), L.minus(Lsaved).normF(),
                    100 * newPrecision, (endTime - startTime) / 1000));

            if (newPrecision < precision){
            	break;
            }

            precision = newPrecision;
            Usaved = U.copy();
            Wsaved = W.copy();
            Lsaved = L.copy();
        }

        U.scale(dropoutZ);
        W.scale(dropoutX);
    }

    /**
     * Computes the precision (correct guesses / nb of tokens) on the devSet with the current parameters U, W, L
     * @param allExamples
     * @return
     */
    public double getPrecision(List<List<Datum>> allExamples, SimpleMatrix U, SimpleMatrix W) {
        double correct_guesses = 0;
        double tokens = 0;
        for (List<Datum> buffer: allExamples) {
            Datum middleWord = buffer.get(windowSize / 2);
            String predictedLabel = predictLabel(buffer, U, W);
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
    public String predictLabel(List<Datum> buffer, SimpleMatrix U, SimpleMatrix W) {
        List<Integer> inputIndex = getLindFromBuffer(buffer);
        SimpleMatrix x = getXFromLind(inputIndex);
        SimpleMatrix xbiased = concatenateWithBias(x);
        SimpleMatrix P = computePFromUWx(U, W, xbiased);
        return labels.get(argmax(P));
    }

    /**
     * Default predict label
     * @param buffer
     * @return
     */
    public String predictLabel(List<Datum> buffer) {
        return predictLabel(buffer, U, W);
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
