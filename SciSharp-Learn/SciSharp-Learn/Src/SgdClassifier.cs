using System;
using static SciSharp_Learn.LinAlgUtils;

namespace SciSharp_Learn
{
    public class SgdClassifier: IClassifier
    {
        // Logistic regression
        // y = 1 / (1 + exp(-xb))
        public double[] BetaParam;
        private readonly int _epochs;
        private readonly double _learningRate;
        
        
        private void Sgd(double[,] x, int[] y)
        {
            Console.WriteLine("Performing Stochastic Gradient Descent.");
            var actual = Array.ConvertAll(y, item => (double)item);
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
            Console.WriteLine("Creating SGD Classifier with no of params:");
            // Initialize beta params randomly
            
            int betaParamsLength = x.Length / y.Length;
            Console.WriteLine(betaParamsLength);
            BetaParam = new double[betaParamsLength];
            Random rand = new Random();
            double paramMax = 1;
            for (int i = 0; i < betaParamsLength; i++)
            {
                BetaParam[i] = rand.Next() % paramMax;
            }
            // Perform Stochastic Gradient Descent
            Sgd(x, y);
        }

        public int[] Predict(double[,] x)
        {
            int resultLength = x.Length / BetaParam.Length;
            double[] regression = Sigmoid(x, BetaParam);
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
            throw new NotImplementedException();
        }
    }
}