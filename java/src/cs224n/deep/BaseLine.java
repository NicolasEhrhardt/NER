
public class BaseLine {
	public static void main(String args[]) throws IOException 
	{
		if(args.lenth < 2)
		{
			 System.out.println("USAGE: java NER ../data/train ../data/dev");
	            return;
		}
		
		BaseLineModel baseLine = new BaseLineModel();
		baseLine.train(args[0]);
		baseLine.test(args[0]);
	}
}