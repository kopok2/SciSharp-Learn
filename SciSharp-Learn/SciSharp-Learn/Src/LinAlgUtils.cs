using System;
using System.Linq;

namespace SciSharp_Learn
{
    public static class LinAlgUtils
    {
        public static double[] DotMatrixVector(double[,] matrix, double[] vector)
        {
            var resultLength = matrix.Length / vector.Length;
            var result = new double[resultLength];
            for (var i = 0; i < resultLength; i++)
            {
                result[i] = 0;
                for (var j = 0; j < vector.Length; j++)
                {
                    result[i] += vector[j] * matrix[i, j];
                }
            }

            return result;
        }
        public static double[] Sigmoid(double[,] x, double[] beta)
        {
            var resultLength = x.Length / beta.Length;
            var result = new double[resultLength];
            var reg = DotMatrixVector(x, beta);
            for (var i = 0; i < resultLength; i++)
            {
                result[i] = 1 / (1 + Math.Exp(-reg[i]));
            }

            return result;
        }

        public static double[,] MatrixTranspose(double[,] x, int len)
        {
            var width = x.Length / len;
            var result = new double[width, len];
            for (var i = 0; i < width; i++)
            {
                for (var j = 0; j < len; j++)
                {
                    result[i, j] = x[j, i];
                }
            }

            return result;
        }
        public static double[] LogisticGradient(double[,] x, double[] beta, double[] y)
        {
            double[] sigmoid = Sigmoid(x, beta);
            double[] a = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                a[i] = sigmoid[i] - y[i];
            }

            return DotMatrixVector(MatrixTranspose(x, y.Length), a);
        }

        public static double MeanSquaredError(double[] prediction, double[] actual)
        {
            return actual.Select((t, i) => Math.Pow((prediction[i] - t), 2)).Sum();
        }
    }
}