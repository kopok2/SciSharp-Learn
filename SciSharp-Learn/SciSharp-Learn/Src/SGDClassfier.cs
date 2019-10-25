using System.Collections.Generic;

namespace SciSharp_Learn
{
    public class SGDClassfier: IClassifier
    {
        // Logistic regression
        // y = 1 / (1 + exp(-xb))
        private double[] beta;
        private double[] b_parameter;

        private double[] DotMatrixVector(double[,] matrix, IReadOnlyList<double> vector)
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
        private double[] Sigmoid(double[,] x)
        {
            double[] result = new double[beta.Length];
            for (int i = 0; i < beta.Length; i++)
            {
                
            }
        }
        private double[] Sgd()
        {
            
        }
        public void Fit(double[,] x, int[] y)
        {
            int epochs = 100, double learningRate = 0.001
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