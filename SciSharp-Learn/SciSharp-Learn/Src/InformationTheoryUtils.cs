using System;

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

        public static double[] ProbabilityDistribution(int[,] x)
        {
            
        }
        public static double InformationGain(int[,]x, int attribute)
        {
            
        }
    }
}