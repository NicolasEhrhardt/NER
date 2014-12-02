package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.ejml.ops.MatrixIO;
import org.ejml.simple.*;


public class FeatureFactory {

    public static final String DOC_START = "-DOCSTART-";
    public static final String START_TOKEN = "<s>";
    public static final String END_TOKEN = "</s>";
    public static final String UNK_TOKEN = "UUUNKKK";


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData = read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData = read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));

        data.add(new Datum(START_TOKEN, "O"));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
            if (line.trim().length() == 0) {
                continue;
            }

            String[] bits = line.split("\\s+");
            String word = bits[0];
            String label = bits[1];

            if (word.equals(DOC_START)) {
                continue;
            }

            word = word.toLowerCase();

            if (word.equals(".")) {
                data.add(new Datum(END_TOKEN, "O"));
                data.add(new Datum(START_TOKEN, "O"));
            } else {
                Datum datum = new Datum(word, label);
                data.add(datum);
            }
		}
        in.close();
        data.add(new Datum(END_TOKEN, "O"));

		return data;
	}

	public static SimpleMatrix allVecs; //access it directly in WindowModel
    /**
     * Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
     * @param vecFilename
     * @return
     * @throws IOException
     */
    public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;
		//set allVecs from filename
        return readMatrixFile(vecFilename, 100232, 50);
    }

    /**
     * Helper to read a matrix from a file
     * @param filename
     * @param numrows
     * @param numcols
     * @return
     * @throws IOException
     */
    public static SimpleMatrix readMatrixFile(String filename, int numrows, int numcols) throws IOException {
        return SimpleMatrix.wrap(MatrixIO.loadCSV(filename, numrows, numcols));
    }

	public static Map<String, Integer> wordToNum = new HashMap<String, Integer>(); //access it directly in WindowModel
	public static Map<Integer, String> numToWord = new HashMap<Integer, String>(); //access it directly in WindowModel
    /**
     * Load vocabulary in an index for efficient lookup.
     * might be useful for word to number lookups
     * @param vocabFilename
     * @return number of words
     * @throws IOException
     */
    public static Map<String, Integer> initializeVocab(String vocabFilename) throws IOException {
        int index = 0;
        BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
            wordToNum.put(word, index);
            numToWord.put(index, word);
            index++;
		}
        in.close();
		return wordToNum;
	}
}
