package cs224n.deep;

import java.io.IOException;
import java.util.List;

public class BaseLine {
	public static void main(String[] args) throws IOException
	{
		if (args.length < 2) {
            System.out.println("USAGE: java -cp classes BaseLine ../data/train ../data/dev");
            return;
        }
		
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
        List<Datum> testData = FeatureFactory.readTestData(args[1]);
        
        BaseLineModel baseline = new BaseLineModel();
        baseline.train(trainData);
        baseline.test(testData);
	}
}
