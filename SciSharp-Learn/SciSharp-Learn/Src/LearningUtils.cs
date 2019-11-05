using System;
using System.Linq;

namespace SciSharp_Learn
{
    public static class LearningUtils
    {
        public static double Accuracy(int[] predicted, int[] actual)
        {
            var correct = actual.Where((t, i) => predicted[i] == t).Count();

            return ((double) correct) / actual.Length;
        }

        public static int[,] DiscreteFilter(double[,] x, int k, int attributeCount)
        {
            int datasetLength = x.Length / attributeCount;
            double[] maxValues = new double[attributeCount];
            double[] minValues = new double[attributeCount];
            double[] valuesWidth = new double[attributeCount];
            for (int i = 0; i < attributeCount; i++)
            {
                maxValues[i] = Double.MinValue;
                minValues[i] = Double.MaxValue;
            }

            for (int i = 0; i < datasetLength; i++)
            {
                for (int j = 0; j < attributeCount; j++)
                {
                    if(x[i, j] > maxValues[j])
                    {
                        maxValues[j] = x[i, j];
                    }

                    if (x[i, j] < minValues[j])
                    {
                        minValues[j] = x[i, j];
                    }
                }
            }

            for (int i = 0; i < attributeCount; i++)
            {
                valuesWidth[i] = maxValues[i] - minValues[i];
            }
            int[,]converted = new int[datasetLength, attributeCount];
            for (int i = 0; i < datasetLength; i++)
            {
                for (int j = 0; j < attributeCount; j++)
                {
                    if (Math.Abs(valuesWidth[j]) > 0.001)
                    {
                        converted[i, j] = Math.Min((int) ((k - 1) * ((x[i, j] - minValues[j]) / valuesWidth[j])), k - 1);
                    }
                    else
                    {
                        converted[i, j] = 0;
                    }
                }
            }
            
            return converted;
        }

        public static void PrintDataset<T>(T[,] dataset, int attributeCount)
        {
            for (int i = 0; i < dataset.Length / attributeCount; i++)
            {
                for (int j = 0; j < attributeCount; j++)
                {
                    Console.Write(dataset[i, j] + " ");
                }
                Console.Write("\n");
            }
        }

        public static void PrintDataset<T>(T[] dataset)
        {
            for (int i = 0; i < dataset.Length; i++)
            {
                Console.Write(dataset[i] + " ");
            }
            Console.Write("\n");
        }
        public static void PrintDataset<T>(T[][][] dataset, int firstDim, int secondDim, int thirdDim)
        {
            for (int i = 0; i < firstDim; i++)
            {
                for (int j = 0; j < secondDim; j++)
                {
                    for (int k = 0; k < thirdDim; k++)
                    {
                        Console.Write(dataset[i][j][k] + " ");
                    }

                    Console.Write("\n");
                }
                Console.Write("\n");
            }
        }
    }
}