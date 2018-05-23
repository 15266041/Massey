package id15266041;

public class Main {

	public static void main(String[] args) 
	{
		ListGenerator knn = new ListGenerator();
		try 
		{
			knn.Run("irisdata.txt");
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}

}
