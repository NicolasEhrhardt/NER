package cs224n.deep;

import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by nicolas on 11/24/14.
 */
public class Utils {

    public static SimpleMatrix concatenateWithBias(SimpleMatrix ... vectors) {
        return concatenateWithBias(Arrays.asList(vectors));
    }

    public static SimpleMatrix concatenate(List<SimpleMatrix> vectors) {
        int size = 0;
        for (SimpleMatrix vector : vectors) {
            size += vector.numRows();
        }
        // one extra for the bias
        SimpleMatrix result = new SimpleMatrix(size, 1);
        int index = 0;
        for (SimpleMatrix vector : vectors) {
            result.insertIntoThis(index, 0, vector);
            index += vector.numRows();
        }
        return result;
    }

    public static SimpleMatrix concatenateWithBias(List<SimpleMatrix> vectors) {
        int size = 0;
        for (SimpleMatrix vector : vectors) {
            size += vector.numRows();
        }
        // one extra for the bias
        size++;
        SimpleMatrix result = new SimpleMatrix(size, 1);
        int index = 0;
        for (SimpleMatrix vector : vectors) {
            result.insertIntoThis(index, 0, vector);
            index += vector.numRows();
        }
        result.set(index, 0, 1.0);
        return result;
    }

    /**
     * Applies softmax to all of the elements of the matrix. The return
     * matrix will have all of its elements sum to 1. If your matrix is
     * not already a vector, be sure this is what you actually want.
     */
    public static SimpleMatrix softmax(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, Math.exp(output.get(i, j)));
            }
        }
        double sum = output.elementSum();
        // will be safe, since exp should never return 0
        return output.scale(1.0 / sum);
    }
    /**
     * Applies log to each of the entries in the matrix. Returns a new matrix.
     */
    public static SimpleMatrix elementwiseApplyLog(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, Math.log(output.get(i, j)));
            }
        }
        return output;
    }
    /**
     * Applies tanh to each of the entries in the matrix. Returns a new matrix.
     */
    public static SimpleMatrix elementwiseApplyTanh(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, Math.tanh(output.get(i, j)));
            }
        }
        return output;
    }
    /**
     * Applies the derivative of tanh to each of the elements in the vector. Returns a new matrix.
     */
    public static SimpleMatrix elementwiseApplyTanhDerivative(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
        output.set(1.0);
        SimpleMatrix tanh = elementwiseApplyTanh(input);
        output = output.minus(tanh.elementMult(tanh));
        return output;
    }

    /**
     * Create an indicator vector of dim n
     * @param n: dimension
     * @param p: position of 1
     */
    public static SimpleMatrix indicator(int n, int p) {
        SimpleMatrix output = new SimpleMatrix(n, 1);
        output.set(p, 1.0);
        return output;
    }

    public static SimpleMatrix withoutLastCol(SimpleMatrix matrix) {
        return matrix.extractMatrix(0, SimpleMatrix.END, 0, matrix.numCols() - 1);
    }

    public static SimpleMatrix perturbateMatrix(SimpleMatrix M, double eps, Random rand) {
        SimpleMatrix noise = SimpleMatrix.random(M.numRows(), M.numCols(), 0, eps, rand);
        return M.plus(noise);
    }
}
