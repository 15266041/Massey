package id15266041;

import java.io.BufferedReader;
import jomp.*;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;

public class KNearestNeighbour 
{
	public void Run() throws Exception
	{
		ArrayList<ArrayList<String>> tests = ReadFile("tests.txt");
		ArrayList<ArrayList<String>> training = ReadFile("training.txt");
		ArrayList<ArrayList<ArrayList<String>>> distances = new ArrayList<ArrayList<ArrayList<String>>>();
		
		for(int i = 0; i < 15; i++)
		{
			ArrayList<ArrayList<String>> pointDistance = new ArrayList<ArrayList<String>>();
			for(int j = 0; j < 5; j++)
			{
				//represents the five shortest distances, each an arraylist with
				//a number (distance) and species.
				String[] init = {"1000", ""};
				pointDistance.add(new ArrayList<String>(Arrays.asList(init)));  
			}
			distances.add(pointDistance);
		}
		
		//omp parrallel
		//omp for
		for(int i = 0; i < 15; i++)
		{
			for(ArrayList<String> tPoint : training)
			{
				float sqrDist = 0;
				for(int j = 0; j <4; j++)
				{
					sqrDist += Math.pow((Float.parseFloat(tPoint.get(j)) - Float.parseFloat(tests.get(i).get(j))), 2);
				}
				
				float distance = (float)Math.sqrt(sqrDist);
				
				for(int j = 0; j < 5; j++)
				{
					if(distance < Float.parseFloat(distances.get(i).get(j).get(0)) && j != 4)
					{

					}
					else
					{
						if(!(distance < Float.parseFloat(distances.get(i).get(j).get(0))))
						{
							j--;
						}
						String[] cd = {Float.toString(distance),tPoint.get(4)};
						ArrayList<String> currentDist = new ArrayList<String>(Arrays.asList(cd)); 
						while (j>=0)
						{
							ArrayList<String> nextDist = distances.get(i).get(j);
							distances.get(i).set(j,currentDist);
							currentDist = nextDist;
							j--;
						}
						break;
					}
				}
			}
		}
	
		
		//check test species
		
		for (ArrayList<ArrayList<String>> d : distances)
		{
			int setosa = 0;
			int versicolor = 0;
			int virginica = 0;
			
			for (ArrayList<String> s : d)
			{
				
				if(s.get(1).equals(" Iris-setosa"))
				{
					setosa++;
				}
				else if (s.get(1).equals(" Iris-versicolor"))
				{
					versicolor++;
				}
				else
				{
					virginica++;
				}
			}
			
			int max = Math.max(setosa, versicolor);
			int newMax = Math.max(max, virginica);
			
			
			
			if (newMax == setosa)
			{
				System.out.println("Setosa");
			}
			else if (newMax == versicolor)
			{
				System.out.println("Versicolor");
			}
			else
			{
				System.out.println("Verginica");
			}
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
