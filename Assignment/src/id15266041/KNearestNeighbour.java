package id15266041;

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

		float[] distances = new float[15*135*2];
		float[] testBuffer = new float[15*5];
		float[] largetraining = new float[135*5];
		int testLoad = 15;
		int trainingLoad = 135/size;
		
		//all load calculations are based on 8 processors
		
		int last = size-1; //rank of last process
		int baseTrainingLoad = 135/size + 1;
		int[] disp = new int[size]; //for scatterv disp int[]
		int[] sendtrain = new int[size]; //for scatterv send int[]
		
		if(135%8 != 0 && rank != last)
		{
			trainingLoad += 1;
		}
		
		float[] trainingBuffer = new float[trainingLoad*5]; //instances of training points for each process
		
		if(rank == 0)
		{
			float[][] tests = ReadFile("tests.txt", 15);
			float[][] training = ReadFile("training.txt", 135);
			
			//Create buffers of one dimension so I can pass them through MPI
			
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
		
		//fill in disp and sendtrain for the scatterv method
		
		for(int i = 0; i<size; i++)
		{
			disp[i] = i*baseTrainingLoad*5;
			if(i!=size-1)
			{
				sendtrain[i] = baseTrainingLoad*5;
			}
			else
			{
				sendtrain[i] = (baseTrainingLoad-1)*5; 
			}
		}
		
		MPI.COMM_WORLD.Scatterv(largetraining, 0, sendtrain, disp, MPI.FLOAT, trainingBuffer, 0,
			baseTrainingLoad*5, MPI.FLOAT, 0);
		
		/*if(rank == 5)
		{
			for(float i : trainingBuffer)
			{
				System.out.println(i);
			}
		}*/
		
		MPI.COMM_WORLD.Bcast(testBuffer, 0, 15*5, MPI.FLOAT, 0); //update all processes testBuffers
		
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
		
		int[] recv = new int[size];
		int[] recvdisp = new int[size];
		for(int i = 0; i < size; i++)
		{
			recvdisp[i] = i * 30 * trainingLoad;
			if(i != size-1)
			{
				recv[i] = 30* baseTrainingLoad;
			}
			else
			{
				recv[i] = 30 * (baseTrainingLoad-1);
			}
		}
		
		MPI.COMM_WORLD.Allgatherv(localDist, 0, 30*trainingLoad, MPI.FLOAT,
				distances, 0, recv, recvdisp, MPI.FLOAT);
		
		if(rank == 1)
		{
			for(int i = 0; i < 15; i++)
			{
				for(int j = 0; j<135; j++)
				{
					for(int k = 0; k<2; k++)
					{
						System.out.println((i*135*2 + j*2 + k) + " " + distances[i*135*2 + j*2 + k]);
					}
				}
			}
			
		}
		
		//find the smallest five distances
		
		int distLoad = 2; //split up the 15 distance groups and scatter to processes
		if(rank == 7)
		{
			distLoad = 1;
		}
		
		float[] dists = new float[distLoad*135*2];
		
		int[] sendtest = new int[] {2*135*2,2*135*2,2*135*2,2*135*2,
				2*135*2,2*135*2,2*135*2,135*2}; //testLoad * trainingnumber * 2
		
		int[] disptest = new int[] {0,135*4,135*8,135*12,135*16,135*20,135*24, 135*28};
		
		MPI.COMM_WORLD.Scatterv(distances, 0, sendtest, disptest, MPI.FLOAT, dists, 0,
				4*135, MPI.FLOAT, 0);
		
		/*
		 * clear up to here
		 */
		
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
					
					if(localdist == Float.POSITIVE_INFINITY)
					{
						float[] cd = {distance, dists[i*2*135 + (j*2) + 1]};
						localFive[i][k][0] = cd[0];
						localFive[i][k][1] = cd[1];
						break;
					}
					//array is in order, largest at [0] and smallest last. 
					if(distance > localdist && k !=4)
					{
						continue;
					}
					
					if(distance == localdist)
					{
						break;
					}
					
					//shift entries to the left by one from where the current distance has been inserted.
					float[] cd = {distance, dists[i*2*135 + (j*2) + 1]}; //current distance 
					while (k<=4)
					{
						float[] nextDist = {localFive[i][k][0], localFive[i][k][1]};
						localFive[i][k][0] = cd[0];
						localFive[i][k][1] = cd[1];
						cd = nextDist;
						k++;
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
				if(s[1] == 1f)
				{
					setosa++;
				}
				else if (s[1] == 2f)
				{
					versicolor++;
				}
				else if (s[1] == 3f)
				{
					virginica++;
				}
			}
			
			int max = Math.max(setosa, versicolor);
			int newMax = Math.max(max, virginica);
			
			if (newMax == setosa)
			{
				System.out.println(rank + " " + testBuffer[(rank + 7*count)*5 + 4] + " = " + "Setosa");
			}
			else if (newMax == versicolor)
			{
				System.out.println(rank + " " + testBuffer[(rank + 7*count)*5 + 4] + " = " + "Versicolor");
			}
			else
			{
				System.out.println(rank + " " + testBuffer[(rank + 7*count)*5 + 4] + " = " + "Verginica");
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
