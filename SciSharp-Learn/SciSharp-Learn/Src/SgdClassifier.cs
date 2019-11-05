using System;
using static System.Array;
using static System.Console;
using static SciSharp_Learn.LinAlgUtils;

namespace SciSharp_Learn
{
    public class SgdClassifier : IClassifier
    {
        // Logistic regression
        // y = 1 / (1 + exp(-xb))
        public double[] BetaParam;
        private readonly int _epochs;
        private readonly double _learningRate;


        private void Sgd(double[,] x, int[] y)
        {
            WriteLine("Performing Stochastic Gradient Descent.");
            var actual = ConvertAll(y, item => (double) item);
            for (var i = 0; i < _epochs; i++)
            {
                var grad = LogisticGradient(x, BetaParam, actual);
                for (var j = 0; j < BetaParam.Length; j++)
                {
                    BetaParam[j] = BetaParam[j] - (_learningRate * grad[j]);
                }
            }
        }

        public SgdClassifier(int epochs = 100, double learningRate = 0.1)
        {
            _epochs = epochs;
            _learningRate = learningRate;
        }

        public void Fit(double[,] x, int[] y)
        {
            WriteLine("Creating SGD Classifier with no of params:");

            // Initialize beta params randomly
            var betaParamsLength = x.Length / y.Length + 1;
            WriteLine(betaParamsLength);
            BetaParam = new double[betaParamsLength];
            var rand = new Random();
            const double paramMax = 1;
            for (var i = 0; i < betaParamsLength; i++)
            {
                BetaParam[i] = rand.Next() % paramMax;
            }

            // Add ones column to data
            var xNew = new double[y.Length, betaParamsLength];
            for (var i = 0; i < y.Length; i++)
            {
                for (var j = 0; j < betaParamsLength - 1; j++)
                {
                    xNew[i, j] = x[i, j];
                }

                xNew[i, betaParamsLength - 1] = 1;
            }

            // Perform Stochastic Gradient Descent
            Sgd(xNew, y);
        }

        public int[] Predict(double[,] x)
        {
            var resultLength = x.Length / (BetaParam.Length - 1);
            // Add ones column to data
            var xNew = new double[resultLength, BetaParam.Length];
            for (var i = 0; i < resultLength; i++)
            {
                for (var j = 0; j < BetaParam.Length - 1; j++)
                {
                    xNew[i, j] = x[i, j];
                }

                xNew[i, BetaParam.Length - 1] = 1;
            }

            var regression = Sigmoid(xNew, BetaParam);
            var result = new int[resultLength];
            for (var i = 0; i < resultLength; i++)
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
    }
}