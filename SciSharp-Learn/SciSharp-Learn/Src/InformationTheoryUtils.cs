using System;
using System.Linq;

namespace SciSharp_Learn
{
    public static class InformationTheoryUtils
    {
        public static double Entropy(double[] probabilities)
        {
            // Entropy = SUM(-pc * log2(pc))
            double result = 0.0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                if (probabilities[i] > 0)
                {
                    result -= probabilities[i] * Math.Log(probabilities[i], 2);
                }
            }

            return Math.Abs(result);
        }

        public static double[] ProbabilityDistribution(int[] y)
        {
            int discreteCount = y.Max() + 1;
            double[] result = new double[discreteCount];
            for (int i = 0; i < y.Length; i++)
            {
                ++result[y[i]];
            }

            for (int i = 0; i < discreteCount; i++)
            {
                result[i] /= y.Length;
            }

            return result;
        }
        public static double InformationGain(int[,]x, int[]y, int attribute)
        {
            
        }
    }
}