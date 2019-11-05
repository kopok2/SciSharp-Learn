using System;
using System.Linq;
using static SciSharp_Learn.LearningUtils;
using static SciSharp_Learn.RegressionTreeStump;

/*
 Module implements State-of-the-Art classification and regression machine learning algorithm -
   Accelerated Gradient Boosting.
   
   Research source:
   link: https://arxiv.org/pdf/1803.02042.pdf
   
   Abstract:
   Gradient tree boosting is a prediction algorithm that sequentially produces a model in the form of
   linear combinations of decision trees, by solving an infinite-dimensional optimization problem.
   We combine gradient boosting and Nesterov’s accelerated descent to design a new algorithm,
   which we callAGB(for Accelerated Gradient Boosting).
   Substantial numerical evidence is provided on both synthetic and real-life data sets
   to assess the excellent performance of the method in a large variety of prediction problems.
   It is empirically shown that AGBis much less sensitive to the shrinkage parameter
   and outputs predictors that are considerably more sparse in the number of trees,
   while retaining the exceptional performance of gradient boosting.

    Implementation by Karol Oleszek 2019
   */
namespace SciSharp_Learn
{
    public class AcceleratedGradientBoostingClassifier : IClassifier
    {
        private readonly int _epochs;
        private readonly double _shrinkage;
        private readonly double[,] _model;
        private readonly double[] _gammaParam;
        private int _paramCount;

        public AcceleratedGradientBoostingClassifier(int epochs, double shrinkage)
        {
            _epochs = epochs;
            _shrinkage = shrinkage;
            _gammaParam = new double[epochs + 1];
            _model = new double[epochs + 1, 5];
        }

        public void Fit(double[,] x, int[] y)
        {
            _paramCount = x.Length / y.Length;
            double lambdaOld = 0;
            double gamma = 1;
            _gammaParam[0] = gamma;
            var mean = (double) y.Sum() / y.Length;
            var gradient = new double[y.Length];
            for (var i = 0; i < y.Length; i++)
            {
                gradient[i] = -mean;
            }


            var regressionSources = RegressDataset(x, y.Length);
            PrintDataset(regressionSources, x.Length / y.Length, 2, y.Length);
            _model[0, 0] = 0;
            _model[0, 1] = 0;
            _model[0, 2] = mean;
            _model[0, 3] = mean;
            _model[0, 4] = 1;
            var testing = 0;
            var zVal = new double[y.Length];
            for (var i = 0; i < y.Length; i++)
            {
                zVal[i] = y[i];
            }

            var loss = double.MaxValue;
            for (var i = 0; i < _epochs; i++)
            {
                Console.WriteLine("Epoch:");
                Console.WriteLine(i + 1);
                Console.WriteLine(zVal.Sum());
                if (zVal.Sum() > loss)
                {
                    break;
                }

                loss = Math.Abs(zVal.Sum());
                // Compute Z
                for (var j = 0; j < y.Length; j++)
                {
                    var sample = new double[x.Length / y.Length];
                    for (var k = 0; k < x.Length / y.Length; k++)
                    {
                        sample[k] = x[j, k];
                    }

                    zVal[j] = y[j] - Infer(sample, i);
                }

                // Fit regression tree
                var yLoc = new double[y.Length];
                for (var j = 0; j < y.Length; j++)
                {
                    yLoc[j] = zVal[(int) regressionSources[testing][1][j]];
                }

                var newTree = Regression(regressionSources[testing][0], yLoc);
                ++testing;
                if (testing >= x.Length / y.Length)
                {
                    testing = 0;
                }

                _model[i + 1, 0] = testing;
                _model[i + 1, 1] = newTree[0];
                _model[i + 1, 2] = newTree[1];
                _model[i + 1, 3] = newTree[2];

                // Update
                var lambdaNew = (1 + Math.Sqrt(1 + 4 * (Math.Pow(lambdaOld, 2)))) / 2;
                gamma = (1 - lambdaOld) / lambdaNew;
                lambdaOld = lambdaNew;
                _gammaParam[i + 1] = gamma;
            }
        }

        public static double[][][] RegressDataset(double[,] x, int dataLength)
        {
            var regressionSources = new double[x.Length / dataLength][][];
            for (var i = 0; i < x.Length / dataLength; i++)
            {
                regressionSources[i] = new double[2][];
                for (var j = 0; j < 2; j++)
                {
                    regressionSources[i][j] = new double[dataLength];
                    if (j == 1)
                    {
                        for (var k = 0; k < dataLength; k++)
                        {
                            regressionSources[i][j][k] = k;
                        }
                    }
                    else
                    {
                        for (var k = 0; k < dataLength; k++)
                        {
                            regressionSources[i][j][k] = x[k, i];
                        }
                    }
                }

                Array.Sort(regressionSources[i][0], regressionSources[i][1]);
            }

            return regressionSources;
        }

        public int[] Predict(double[,] x)
        {
            var result = new int[x.Length / _paramCount];
            for (var j = 0; j < x.Length / _paramCount; j++)
            {
                var sample = new double[_paramCount];
                for (var k = 0; k < _paramCount; k++)
                {
                    sample[k] = x[j, k];
                }

                result[j] = (int) Math.Round(Infer(sample, _epochs));
            }

            return result;
        }

        public double[] Regress(double[,] x)
        {
            var result = new double[x.Length / _paramCount];
            for (var j = 0; j < x.Length / _paramCount; j++)
            {
                var sample = new double[_paramCount];
                for (var k = 0; k < _paramCount; k++)
                {
                    sample[k] = x[j, k];
                }

                result[j] = Infer(sample, _epochs);
            }

            return result;
        }

        private double Infer(double[] sample, int inferBound, bool useG = false)
        {
            if (sample == null) throw new ArgumentNullException(nameof(sample));
            var ft = _model[0, 2];
            var gt = _model[0, 2];
            double ft1 = 0;
            double gt1 = 0;
            for (var i = 0; i < inferBound; i++)
            {
                var treeReg = sample[(int) _model[i + 1, 0]] < _model[i + 1, 1] ? _model[i + 1, 2] : _model[i + 1, 3];
                ft1 = gt + _shrinkage * treeReg;
                gt1 = (1 - _gammaParam[i + 1]) * ft1 + _gammaParam[i + 1] * ft;
                gt = gt1;
                ft = ft1;
            }

            return !useG ? ft1 : gt1;
        }
    }
}