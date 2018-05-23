package id15266041;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public class KNearestNeighbour 
{
	public void Run() throws Exception
	{
		ArrayList<ArrayList<String>> tests = ReadFile("tests.txt");
		ArrayList<ArrayList<String>> training = ReadFile("training.txt");
		
		for(ArrayList<String> test : tests)
		{
			
		}
	}
	
	private  ArrayList<ArrayList<String>> ReadFile(String filename) throws Exception
	{
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);
		String line;
		ArrayList<ArrayList<String>> data = new ArrayList<ArrayList<String>>();
		
		while((line = br.readLine()) != null)
		{
			ArrayList<String> input = new ArrayList<String>(Arrays.asList(line.split(","))); //comma seperated data
			if(input.size() > 1)
			{
				data.add(input);
			}
		}
		br.close();
		return data;
	}
}
