package cs224n.deep;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

public class BaseLineModel {
	
	HashMap<String, String> ners;
	
	public BaseLineModel()
	{
		ners = new HashMap<String,String>();
	}
	
	public void train(List<Datum> data)
	{
		System.out.println("-- Testing data --");
		for(Datum dat : data)
		{
			String word = dat.word;
			String label = dat.label;
			ners.put(word, label);
		}
	}
	
	public void test(List<Datum> data) throws IOException
	{
		System.out.println("-- Training data --");
		FileWriter fw = new FileWriter("baseline_prediction.out");
		for(Datum dat : data)
		{
			String word = dat.word;
			String label = dat.label;
			String predLabel = ners.get(word);
			if(predLabel == null)
			{
				predLabel = "O";
			}
			fw.write(word + '\t');
			fw.write(label + '\t');
			fw.write(predLabel + '\n');
		}
		fw.close();
	}
}
