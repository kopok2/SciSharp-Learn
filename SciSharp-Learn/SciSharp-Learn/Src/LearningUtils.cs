using System.Collections.Generic;
using System.Linq;
using static System.Console;
using static System.Math;

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
            var datasetLength = x.Length / attributeCount;
            var maxValues = new double[attributeCount];
            var minValues = new double[attributeCount];
            var valuesWidth = new double[attributeCount];
            for (var i = 0; i < attributeCount; i++)
            {
                maxValues[i] = double.MinValue;
                minValues[i] = double.MaxValue;
            }

            for (var i = 0; i < datasetLength; i++)
            {
                for (var j = 0; j < attributeCount; j++)
                {
                    if (x[i, j] > maxValues[j])
                    {
                        maxValues[j] = x[i, j];
                    }

                    if (x[i, j] < minValues[j])
                    {
                        minValues[j] = x[i, j];
                    }
                }
            }

            for (var i = 0; i < attributeCount; i++)
            {
                valuesWidth[i] = maxValues[i] - minValues[i];
            }

            var converted = new int[datasetLength, attributeCount];
            for (var i = 0; i < datasetLength; i++)
            {
                for (var j = 0; j < attributeCount; j++)
                {
                    if (Abs(valuesWidth[j]) > 0.001)
                    {
                        converted[i, j] = Min((int) ((k - 1) * ((x[i, j] - minValues[j]) / valuesWidth[j])),
                            k - 1);
                    }
                    else
                    {
                        converted[i, j] = 0;
                    }
                }
            }

            return converted;
        }

        public static void PrintDataset<T>(IEnumerable<T> dataset)
        {
            foreach (var t in dataset)
            {
                Write(t + " ");
            }

            Write("\n");
        }

        public static void PrintDataset<T>(T[][][] dataset, int firstDim, int secondDim, int thirdDim)
        {
            for (var i = 0; i < firstDim; i++)
            {
                for (var j = 0; j < secondDim; j++)
                {
                    for (var k = 0; k < thirdDim; k++)
                    {
                        Write(dataset[i][j][k] + " ");
                    }

                    Write("\n");
                }

                Write("\n");
            }
        }
    }
}