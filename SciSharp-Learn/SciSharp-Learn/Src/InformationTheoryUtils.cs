using System;
using System.Collections.Generic;
using System.Linq;

namespace SciSharp_Learn
{
    public static class InformationTheoryUtils
    {
        public static double Entropy(double[] probabilities)
        {
            // Entropy = SUM(-pc * log2(pc))
            double result = probabilities.Where(t => t > 0).Aggregate(0.0, (current, t) => current - t * Math.Log(t, 2));

            return Math.Abs(result);
        }

        public static double[] ProbabilityDistribution(int[] y)
        {
            double[] result;
            if(y.Length > 0)
            {
                int discreteCount = y.Max() + 1;
                result = new double[discreteCount];
                foreach (var t in y)
                {
                    ++result[t];
                }
    
                for (int i = 0; i < discreteCount; i++)
                {
                    result[i] /= y.Length;
                }
            }
            else
            {
                result = new double[] {0};
            }

            return result;
        }
        public static double InformationGain(int[,]x, int[]y, int attribute)
        {
            double baseEntropy = Entropy(ProbabilityDistribution(y));
            double newEntropy = 0;
            
            // Calculate attribute states
            int attributeStateCount = 0;
            for (int i = 0; i < y.Length; i++)
            {
                if (x[i, attribute] > attributeStateCount)
                {
                    attributeStateCount = x[i, attribute];
                }
            }
            ++attributeStateCount;
            
            // Count attribute state occurence
            int[] attributeStateOccurenceCount = new int[attributeStateCount];
            for (int i = 0; i < y.Length; i++)
            {
                ++attributeStateOccurenceCount[x[i, attribute]];
            }

            // Count probability distribution estimates
            int[][] probabilities = new int[attributeStateCount][];
            for (int i = 0; i < attributeStateCount; i++)
            {
                probabilities[i] = new int[attributeStateOccurenceCount[i]];
            }
            int[] probabilitiesFillCount = new int[attributeStateCount];
            for (int i = 0; i < y.Length; i++)
            {
                probabilities[x[i, attribute]][probabilitiesFillCount[x[i, attribute]++]] = y[i];
            }

            for (int i = 0; i < attributeStateCount; i++)
            {
                newEntropy += ((double)attributeStateOccurenceCount[i] / y.Length) * Entropy(ProbabilityDistribution(probabilities[i]));
            }
            
            return baseEntropy - newEntropy;
        }

        public static int BestAttribute(int[,] x, int[] y, List<int> attributes)
        {
            int attributeCount = x.Length / y.Length;
            double maxInformationGain = 0;
            int resultAttribute = -1;
            for (int i = 0; i < attributeCount; i++)
            {
                if (attributes.Contains(i))
                {
                    double attributeInformationGain = InformationGain(x, y, i);
                    if (attributeInformationGain > maxInformationGain)
                    {
                        maxInformationGain = attributeInformationGain;
                        resultAttribute = i;
                    }
                }
            }

            return resultAttribute;
        }
    }
}