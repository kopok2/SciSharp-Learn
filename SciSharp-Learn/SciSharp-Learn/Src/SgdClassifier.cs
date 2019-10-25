using System;
using System.Collections.Generic;
using System.Linq;

namespace SciSharp_Learn
{
    public class SgdClassifier: IClassifier
    {
        // Logistic regression
        // y = 1 / (1 + exp(-xb))
        private double[] betaParam;
        private int epochs;
        private double learningRate;
        
        public static double[] DotMatrixVector(double[,] matrix, IReadOnlyList<double> vector)
        {
            var resultLength = matrix.Length / vector.Count;
            var result = new double[resultLength];
            for (var i = 0; i < resultLength; i++)
            {
                result[i] = 0;
                for (var j = 0; j < vector.Count; j++)
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
            var reg = SgdClassifier.DotMatrixVector(x, beta);
            for (var i = 0; i < resultLength; i++)
            {
                result[i] = 1 / (1 + Math.Exp(-reg[i]));
            }

            return result;
        }

        public static double[,] MatrixTranspose(double[,] x)
        {
            var n = (int) Math.Sqrt(x.Length);
            var result = new double[n,n];
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    result[i, j] = x[j, i];
                }
            }

            return result;
        }
        public static double[] LogisticGradient(double[,] x, double[] beta, double[] y)
        {
            double[] sigmoid = SgdClassifier.Sigmoid(x, beta);
            double[] a = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                a[i] = sigmoid[i] - y[i];
            }

            return SgdClassifier.DotMatrixVector(SgdClassifier.MatrixTranspose(x), a);
        }

        public static double MeanSquaredError(double[] prediction, double[] actual)
        {
            return actual.Select((t, i) => Math.Pow((prediction[i] - t), 2)).Sum();
        }
        private double[] Sgd()
        {
            throw new System.NotImplementedException();
        }

        public SgdClassifier(int aepochs = 100, double alearningRate = 0.001)
        {
            epochs = aepochs;
            learningRate = alearningRate;
        }
        public void Fit(double[,] x, int[] y)
        {
            
        }

        public int[] Predict(double[,] x)
        {
            throw new System.NotImplementedException();
        }

        public double Score()
        {
            throw new System.NotImplementedException();
        }
    }
}