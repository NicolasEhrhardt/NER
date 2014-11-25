package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.simple.*;

import static cs224n.deep.Utils.*;

public class WindowModel {

    protected SimpleMatrix L, W, U;
    public int windowSize, wordSize, hiddenSize, numWords, K;
    public double lr;
    public List<String> labels;

    public Map<String, Integer> wordToNum;

    public WindowModel(int windowSize, int wordSize, int hiddenSize, double lr,
                       Map<String, Integer> wordToNum, List<String> labels){
        this.windowSize = windowSize;
        this.wordSize = wordSize;
        this.hiddenSize = hiddenSize;
        this.lr = lr;
        this.wordToNum = wordToNum;
        this.numWords = wordToNum.size();
        this.labels = labels;
        this.K = labels.size();
    }

    public void loadVocab(SimpleMatrix allVec) {
        this.L = allVec;
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


    // Compute Error

    private static SimpleMatrix computeError(SimpleMatrix y, SimpleMatrix p) {
        return y.minus(p);
    }

    // Compute delta

    private static SimpleMatrix computeDelta(SimpleMatrix errorVector, SimpleMatrix Utruncated, SimpleMatrix z) {
        return elementwiseApplyTanhDerivative(z).elementMult(Utruncated.transpose().mult(errorVector));
    }

    // Compute P

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
            SimpleMatrix hbiased = concatenateWithBias(SimpleMatrix.random(hiddenSize, 1, 0, 1, rand));
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);
            SimpleMatrix eps = SimpleMatrix.random(Utest.numRows(), Utest.numCols(), 0, precision, rand);
            SimpleMatrix Ueps = Utest.plus(eps);

            SimpleMatrix trueGrad = computeUgradFromUh(y, Utest, hbiased);
            double check = trueGrad.elementMult(eps).elementSum();
            double diff = computeCostFromUh(y, Ueps, hbiased) - computeCostFromUh(y, Utest, hbiased);

            differences.add(Math.abs(check - diff) / precision);
        }
        return Collections.max(differences);
    }

    // W grad

    private SimpleMatrix computeWgradFromDelta(SimpleMatrix delta, SimpleMatrix xbiased) {
        return delta.mult(xbiased.transpose());
    }

    private SimpleMatrix computeWgradFromUWx(SimpleMatrix y, SimpleMatrix U, SimpleMatrix W, SimpleMatrix xbiased) {
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
            SimpleMatrix x = concatenateWithBias(SimpleMatrix.random(windowSize * wordSize, 1, 0, 1, rand));
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix Wtest = helperInitWeights(W.numCols(), W.numRows(), rand);
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);
            SimpleMatrix eps = SimpleMatrix.random(Wtest.numRows(), Wtest.numCols(), 0, precision, rand);
            SimpleMatrix Weps = Wtest.plus(eps);

            SimpleMatrix trueGrad = computeWgradFromUWx(y, Utest, Wtest, x);
            double check = trueGrad.elementMult(eps).elementSum();
            double diff = computeCostFromUWx(y, Utest, Weps, x) - computeCostFromUWx(y, Utest, Wtest, x);

            differences.add(Math.abs(check - diff) / precision);
        }
        return Collections.max(differences);
    }

    // X gradient

    private SimpleMatrix computeXgradFromDelta(SimpleMatrix delta, SimpleMatrix Wtruncated) {
        return Wtruncated.transpose().mult(delta);
    }

    private SimpleMatrix computeXgradFromUWx(SimpleMatrix y, SimpleMatrix U, SimpleMatrix W, SimpleMatrix xtruncated) {
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
            SimpleMatrix xtest = SimpleMatrix.random(windowSize * wordSize, 1, 0, 1, rand);
            SimpleMatrix y = indicator(K, rand.nextInt(K));
            SimpleMatrix Wtest = helperInitWeights(W.numCols(), W.numRows(), rand);
            SimpleMatrix Utest = helperInitWeights(U.numCols(), U.numRows(), rand);
            SimpleMatrix eps = SimpleMatrix.random(xtest.numRows(), xtest.numCols(), 0, precision, rand);
            SimpleMatrix xeps = xtest.plus(eps);

            SimpleMatrix trueGrad = computeXgradFromUWx(y, Utest, Wtest, xtest);
            double check = trueGrad.elementMult(eps).elementSum();
            double diff = computeCostFromUWx(y, Utest, Wtest, concatenateWithBias(xeps)) -
                    computeCostFromUWx(y, Utest, Wtest, concatenateWithBias(xtest));

            differences.add(Math.abs(check - diff) / precision);
        }
        return Collections.max(differences);
    }

    // Processing helpers

    private int getWordIndex(String word) {
        if (wordToNum.containsKey(word)) {
            return wordToNum.get(word);
        } else {
            return wordToNum.get(FeatureFactory.UNK_TOKEN);
        }
    }

    private void updateWeights(List<Datum> buffer) {
        List<Integer> inputIndex = new ArrayList<Integer>();

        String label = "O";
        int pos = 0;
        for (Datum datum: buffer) {
            // store label of middle of buffer
            if (pos == windowSize / 2) {
                label = datum.label;
            }

            // get vector representation
            int index = getWordIndex(datum.word);
            inputIndex.add(index);
            pos++;
        }

        List<SimpleMatrix> inputVectors = new ArrayList<SimpleMatrix>();
        for (int index: inputIndex) {
            SimpleMatrix curVec = L.extractVector(true, index);
            inputVectors.add(curVec.transpose());
        }

        SimpleMatrix x = concatenate(inputVectors);
        SimpleMatrix xbiased = concatenateWithBias(x);
        SimpleMatrix z = W.mult(xbiased);
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
        U.set(U.plus(lr, Ugrad));

        // Update W
        W.set(W.plus(lr, Wgrad));


        // Update x -> L
        x.set(x.plus(lr, Xgrad));

        for (int i = 0; i < inputIndex.size(); i++) {
            int index = inputIndex.get(i);
            L.insertIntoThis(index, 0, x.extractMatrix(i * wordSize, (i + 1) * wordSize, 0, 1).transpose());
        }
    }

    /**
     * Simplest SGD training
     */
    public void train(List<Datum> _trainData ){

        int count = 0;
        SimpleMatrix Usaved = U.copy();
        SimpleMatrix Wsaved = W.copy();
        SimpleMatrix Lsaved = L.copy();

        for (int epoch = 0; epoch < 10; epoch++) {
            List<Datum> buffer = new ArrayList<Datum>();
            for (Datum datum : _trainData) {

                // Clear buffer if we are restartin a sentence
                if (datum.label.equals(FeatureFactory.START_TOKEN)) {
                    buffer.clear();
                }

                // Add token if we haven't reached the window size
                if (buffer.size() < windowSize) {
                    buffer.add(datum);
                }

                // If the buffer is the right size, update the weights and remove the oldest token
                if (buffer.size() == windowSize) {
                    updateWeights(buffer);
                    count++;
                    buffer.remove(0);
                }

                if ((count % 1000) == 0) {
                    System.out.println(String.format("Viewd %d, delta U %f, delta W %f, delta L %f",
                            count, U.minus(Usaved).normF(), W.minus(Wsaved).normF(), L.minus(Lsaved).normF()));
                    Usaved = U.copy();
                    Wsaved = W.copy();
                    Lsaved = L.copy();
                }
            }
        }
    }

    public void test(List<Datum> testData){
        // TODO
    }

}
