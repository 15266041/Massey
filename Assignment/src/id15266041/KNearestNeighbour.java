package parallel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import mpi.MPI;

public class KNearestNeighbour 
{
	public void Run(String[] args) throws Exception
	{
		MPI.Init(args);
		int rank = MPI.COMM_WORLD.Rank();
		int size = MPI.COMM_WORLD.Size();

		float[] trainingBuffer = new float[170];
		float[] distances = new float[15*135*2];
		float[] testBuffer = new float[15*5];
		float[] largetraining = new float[135*5];
		int testLoad = 15;
		int trainingLoad = 34;

		if(rank == 3)
		{
			trainingLoad = 33;
		}
		
		if(rank == 0)
		{
			float[][] tests = ReadFile("tests.txt", 15);
			float[][] training = ReadFile("training.txt", 135);
			
			//Create buffers of one dimention so I can pass them through MPI
			
			for(int i = 0; i < 15; i++)
			{
				for(int j = 0; j < 5; j++)
				{
					testBuffer[(i*5)+j] = tests[i][j];
				}
			}
			
			for(int i = 0; i < 135; i++)
			{
				for(int j = 0; j < 5; j++)
				{
					largetraining[i*5+j] = training[i][j];
				}
			}
			
		}
		
		int[] disp = new int[] {0,34*5,34*10,34*15};
		int[] sendtrain = new int[] {34*5,34*5,34*5,33*5};
		
		MPI.COMM_WORLD.Scatterv(largetraining, 0, sendtrain, disp, MPI.FLOAT, trainingBuffer, 0,
			34*5, MPI.FLOAT, 0);
		
		float[] localDist = new float[testLoad*trainingLoad*2];
		
		//indexing, i represents the test point, j the training point and k a distance plant type pair.
		
		
		for(int i = 0; i < testLoad; i++)
		{
			for(int j = 0; j <trainingLoad; j++)
			{
				float sqrDist = 0;
				for(int k = 0; k <4; k++)
				{
					sqrDist += Math.pow(trainingBuffer[j*5+k] - testBuffer[i*5+k], 2);
				}
				
				float distance = (float)Math.sqrt(sqrDist);
				localDist[(i*2*trainingLoad)+(j*2)] = distance; //add point-point distance
				localDist[(i*2*trainingLoad) + (j*2)+1] = trainingBuffer[j*5 + 4]; //add flower type
			}
		}

		int[] recv = new int[]{15*34*2,15*34*2,
				15*34*2,15*33*2};
		
		int[] recvdisp = new int[] {0, 30*34, 60*34, 90*34};
		
		MPI.COMM_WORLD.Allgatherv(localDist, 0, 15*trainingLoad*2, MPI.FLOAT,
				distances, 0, recv, recvdisp, MPI.FLOAT);
	
		/*
		 * Clear up to here ")
		 * 
		 */
		//find the smallest five distances
		
		int distLoad = 4; //split up the 15 distance groups and scatter to processes
		if(rank == 3)
		{
			distLoad = 3;
		}
		
		float[] dists = new float[distLoad*135*2];
		
		int[] sendtest = new int[] {4*135*2,4*135*2,4*135*2,3*135*2};
		int[] disptest = new int[] {0,distLoad*135*2,distLoad*135*4,distLoad*135*6};
		
		MPI.COMM_WORLD.Scatterv(distances, 0, sendtest, disptest, MPI.FLOAT, dists, 0,
				4*135*2, MPI.FLOAT, 0);
		
		float[][][] localFive = new float[distLoad][5][2];
		
		for(int i = 0; i < distLoad; i++) //each test
		{
			for(int j = 0; j < 5; j++ )
			{
				localFive[i][j] = new float[]{Float.POSITIVE_INFINITY, 0f}; //fill local five with first 5 instances
			}
			
			for(int j = 0; j < 135; j++)
			{
				float distance = dists[i*2*135 + (j*2)];
				for(int k = 0; k <5; k++)
				{
					float localdist = localFive[i][k][0];
					
					//array is in order, largest at [0] and smallest last. 
					if(distance < localdist && k !=4)
					{
						continue;
					}
					
					if(distance >= localdist)
					{
						k--;
					}
					
					//shift entries to the left by one from where the current distance has been inserted.
					float[] cd = {distance, dists[i*2*135 + (j*2) + 1]}; //current distance 
					while (k>=0)
					{
						float[] nextDist = {localFive[i][k][0], localFive[i][k][1]};
						localFive[i][k][0] = cd[0];
						localFive[i][k][1] = cd[1];
						cd = nextDist;
						k--;
					}
					break;
				}
			}
		}
		
		//check test species
		int count = 0;
		for (float[][] d : localFive)
		{
			int setosa = 0;
			int versicolor = 0;
			int virginica = 0;
			
			for (float[] s : d)
			{
				//System.out.println(s[1]);
				if(s[1] == 1f)
				{
					setosa++;
				}
				else if (s[1] == 2f)
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
				System.out.println(testBuffer[count*5+4] + " = " + "Setosa");
			}
			else if (newMax == versicolor)
			{
				System.out.println(testBuffer[count*5+4] + " = " + "Versicolor");
			}
			else
			{
				System.out.println(testBuffer[count*5+4] + " = " + "Verginica");
			}
			count++;
		}
	}
	
	private  float[][] ReadFile(String filename, int size) throws Exception
	{
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr);
		String line;
		float[][] data = new float[size][5];
		int count = 0;
		while((line = br.readLine()) != null)
		{
			String[] input = line.split(","); //comma seperated data
			if(input[0] != "")
			{
				//convert the values to float and the species value to a float id. This allows for float
				//usage in MPI methods.
				float[] converted = new float[5];
				for(int i = 0; i < 5; i++)
				{
					if(input[i].equals(" Iris-setosa"))
					{
						converted[i] = 1f;
						break;
					}
					else if (input[i].equals(" Iris-versicolor"))
					{
						converted[i] = 2f;
						break;
					}
					else if (input[i].equals(" Iris-virginica"))
					{
						converted[i] = 3f;
						break;
					}
					converted[i] = Float.parseFloat(input[i]);
				}
				data[count] = converted;
			}
			count++;
		}
		br.close();
		return data;
	}
}
