using System;
using System.Collections.Generic;
using System.Linq;

namespace SciSharp_Learn
{
    public class SgdClassifier: IClassifier
    {
        // Logistic regression
        // y = 1 / (1 + exp(-xb))
        public const bool verbose = false;
        public double[] betaParam;
        private int epochs;
        private double learningRate;
        
        public static double[] DotMatrixVector(double[,] matrix, double[] vector)
        {
            var resultLength = matrix.Length / vector.Length;
            if (verbose)
            {
                System.Console.WriteLine("Multiplying matrix with vector:");
                System.Console.WriteLine(matrix.Length);
                System.Console.WriteLine(vector.Length);
                System.Console.WriteLine(resultLength);
            }
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
            var reg = SgdClassifier.DotMatrixVector(x, beta);
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
            double[] sigmoid = SgdClassifier.Sigmoid(x, beta);
            double[] a = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                a[i] = sigmoid[i] - y[i];
            }

            return SgdClassifier.DotMatrixVector(SgdClassifier.MatrixTranspose(x, y.Length), a);
        }

        public static double MeanSquaredError(double[] prediction, double[] actual)
        {
            return actual.Select((t, i) => Math.Pow((prediction[i] - t), 2)).Sum();
        }
        private void Sgd(double[,] x, int[] y)
        {
            System.Console.WriteLine("Performing Stochastic Gradient Descent.");
            double[] prediction = Array.ConvertAll(this.Predict(x), item => (double)item);
            double[] actual = Array.ConvertAll(y, item => (double)item);
            double cost = SgdClassifier.MeanSquaredError(prediction, actual);
            double costStep = 1;
            double oldCost;
            double[] grad;
            for (int i = 0; i < epochs; i++)
            {
                oldCost = cost;
                grad = SgdClassifier.LogisticGradient(x, this.betaParam, actual);
                for (int j = 0; j < this.betaParam.Length; j++)
                {
                    this.betaParam[j] = this.betaParam[j] - (this.learningRate * grad[j]);
                }
                prediction = Array.ConvertAll(this.Predict(x), item => (double)item);
                cost = SgdClassifier.MeanSquaredError(prediction, actual);
                costStep = oldCost - cost;
                if (verbose)
                {
                    System.Console.WriteLine("Performing stochastic gradient descent step:");
                    System.Console.WriteLine(i);
                    System.Console.WriteLine(costStep);
                    System.Console.WriteLine(cost);
                    foreach(var item in this.betaParam)
                    {
                        Console.WriteLine(item.ToString());
                    }
                }
            }
        }

        public SgdClassifier(int aepochs = 100, double alearningRate = 0.1)
        {
            epochs = aepochs;
            learningRate = alearningRate;
        }
        public void Fit(double[,] x, int[] y)
        {
            System.Console.WriteLine("Creating SGD Classifier with no of params:");
            // Initialize beta params randomly
            
            int betaParamsLength = x.Length / y.Length;
            System.Console.WriteLine(betaParamsLength);
            this.betaParam = new double[betaParamsLength];
            Random rand = new Random();
            double paramMax = 1;
            for (int i = 0; i < betaParamsLength; i++)
            {
                this.betaParam[i] = rand.Next() % paramMax;
            }
            // Perform Stochastic Gradient Descent
            this.Sgd(x, y);
        }

        public int[] Predict(double[,] x)
        {
            int resultLength = x.Length / this.betaParam.Length;
            double[] regression = SgdClassifier.Sigmoid(x, this.betaParam);
            int[] result = new int[resultLength];
            for (int i = 0; i < resultLength; i++)
            {
                if (regression[i] >= 0.5)
                {
                    result[i] = 1;
                }
                else
                {
                    result[i] = 0;
                }
            }

            return result;
        }

        public double Score()
        {
            throw new System.NotImplementedException();
        }
    }
}