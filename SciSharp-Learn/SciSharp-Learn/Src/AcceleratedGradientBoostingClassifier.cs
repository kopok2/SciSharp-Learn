using System.Linq;

namespace SciSharp_Learn
{
    public class AcceleratedGradientBoostingClassifier : IClassifier
    {
        private int _epochs;
        private double _shrinkage;
        private double[,] _model;

        public AcceleratedGradientBoostingClassifier(int epochs, double shrinkage)
        {
            _epochs = epochs;
            _shrinkage = shrinkage;
            _model = new double[epochs,4];
        }

        public void Fit(double[,] x, int[] y)
        {
            double lambdaNew = 0;
            double lambdaOld = 0;
            double gamma = 1;
            double mean = (double)y.Sum() / y.Length;
            var gradient = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                gradient[i] = mean;
            }

            _model[0, 0] = 0;
            _model[0, 1] = 0;
            _model[0, 2] = mean;
            _model[0, 3] = mean;
            int testing = 0;
            double[]zVal = new double[y.Length];
            for (int i = 0; i < _epochs; i++)
            {
                
            }

        }

        public int[] Predict(double[,] x)
        {
            throw new System.NotImplementedException();
        }
    }
}