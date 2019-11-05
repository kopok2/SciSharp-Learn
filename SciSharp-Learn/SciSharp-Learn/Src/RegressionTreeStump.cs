using System;

namespace SciSharp_Learn
{
    public static class RegressionTreeStump
    {
        public static double[] Regression(double[] x, double[] y)
        {
            var loss = double.MaxValue;
            var result = new double[3];
            for (var boundary = 1; boundary < y.Length; boundary++)
            {
                var threshold = (x[boundary - 1] + x[boundary]) / 2;
                double sumLeft = 0;
                double sumRight = 0;
                for (var i = 0; i < boundary; i++)
                {
                    sumLeft += y[i];
                }

                for (var i = boundary; i < y.Length; i++)
                {
                    sumRight += y[i];
                }

                var regLeft = sumLeft / boundary;
                var regRight = sumRight / (y.Length - boundary);
                double newLoss = 0;
                for (var i = 0; i < y.Length; i++)
                {
                    if (i < boundary)
                    {
                        newLoss += Math.Pow((y[i] - regLeft), 2);
                    }
                    else
                    {
                        newLoss += Math.Pow((y[i] - regRight), 2);
                    }
                }

                if (!(newLoss < loss)) continue;
                loss = newLoss;
                result[0] = threshold;
                result[1] = regLeft;
                result[2] = regRight;
            }

            return result;
        }
    }
}