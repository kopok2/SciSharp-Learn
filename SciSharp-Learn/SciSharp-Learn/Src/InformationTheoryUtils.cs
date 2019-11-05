using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace SciSharp_Learn
{
    public static class InformationTheoryUtils
    {
        public static double Entropy(double[] probabilities)
        {
            if (probabilities == null) throw new ArgumentNullException(nameof(probabilities));
            // Entropy = SUM(-pc * log2(pc))
            var result = probabilities.Where(t => t > 0).Aggregate(0.0, (current, t) => current - t * Log(t, 2));

            return Abs(result);
        }

        public static double[] ProbabilityDistribution(int[] y)
        {
            double[] result;

            if (y.Length > 0)
            {
                var discreteCount = y.Max() + 1;
                result = new double[discreteCount];
                foreach (var t in y)
                {
                    ++result[t];
                }

                for (var i = 0; i < discreteCount; i++)
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

        public static double InformationGain(int[,] x, int[] y, int attribute)
        {
            var baseEntropy = Entropy(ProbabilityDistribution(y));
            double newEntropy = 0;

            // Calculate attribute states
            var attributeStateCount = 0;
            for (var i = 0; i < y.Length; i++)
            {
                if (x[i, attribute] > attributeStateCount)
                {
                    attributeStateCount = x[i, attribute];
                }
            }

            ++attributeStateCount;

            // Count attribute state occurence
            var attributeStateOccurenceCount = new int[attributeStateCount];
            for (var i = 0; i < y.Length; i++)
            {
                ++attributeStateOccurenceCount[x[i, attribute]];
            }

            // Count probability distribution estimates
            var probabilities = new int[attributeStateCount][];
            for (var i = 0; i < attributeStateCount; i++)
            {
                probabilities[i] = new int[attributeStateOccurenceCount[i]];
            }

            var probabilitiesFillCount = new int[attributeStateCount];
            for (var i = 0; i < y.Length; i++)
            {
                probabilities[x[i, attribute]][probabilitiesFillCount[x[i, attribute]++]] = y[i];
            }

            for (var i = 0; i < attributeStateCount; i++)
            {
                newEntropy += ((double) attributeStateOccurenceCount[i] / y.Length) *
                              Entropy(ProbabilityDistribution(probabilities[i]));
            }

            return baseEntropy - newEntropy;
        }

        public static int BestAttribute(int[,] x, int[] y, List<int> attributes)
        {
            var attributeCount = x.Length / y.Length;
            double maxInformationGain = 0;
            var resultAttribute = -1;
            for (var i = 0; i < attributeCount; i++)
            {
                if (!attributes.Contains(i)) continue;
                var attributeInformationGain = InformationGain(x, y, i);
                if (!(attributeInformationGain > maxInformationGain)) continue;
                maxInformationGain = attributeInformationGain;
                resultAttribute = i;
            }

            return resultAttribute;
        }
    }
}