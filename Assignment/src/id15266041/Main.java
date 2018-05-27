package parallel;

public class Main {

	public static void main(String[] args) 
	{
		KNearestNeighbour knn = new KNearestNeighbour();
		try 
		{
			knn.Run(args);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}

}
