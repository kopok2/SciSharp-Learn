using System;
using System.Linq;
using System.Management.Instrumentation;
using static SciSharp_Learn.LearningUtils;
using static SciSharp_Learn.RegressionTreeStump;

namespace SciSharp_Learn
{
    public class AcceleratedGradientBoostingClassifier : IClassifier
    {
        private int _epochs;
        private double _shrinkage;
        private double[,] _model;
        private double[] _gamma_param;
        private int _paramCount;

        public AcceleratedGradientBoostingClassifier(int epochs, double shrinkage)
        {
            _epochs = epochs;
            _shrinkage = shrinkage;
            _gamma_param = new double[epochs + 1];
            _model = new double[epochs + 1, 5];
        }

        public void Fit(double[,] x, int[] y)
        {
            _paramCount = x.Length / y.Length;
            double lambdaNew = 0;
            double lambdaOld = 0;
            double gamma = 1;
            _gamma_param[0] = gamma;
            double mean = (double) y.Sum() / y.Length;
            var gradient = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                gradient[i] = -mean;
            }


            double[][][] regressionSources = RegressDataset(x, y.Length);
            PrintDataset(regressionSources, x.Length / y.Length, 2, y.Length);
            _model[0, 0] = 0;
            _model[0, 1] = 0;
            _model[0, 2] = mean;
            _model[0, 3] = mean;
            _model[0, 4] = 1;
            int testing = 0;
            double[] zVal = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                zVal[i] = y[i];
            }
            for (int i = 0; i < _epochs; i++)
            {
                Console.WriteLine("Epoch:");
                Console.WriteLine(i + 1);
                PrintDataset(zVal);
                PrintDataset(gradient);
                //PrintDataset(_model, 5);
                // Compute Z
                for (int j = 0; j < y.Length; j++)
                {
                    double[] sample = new double[x.Length / y.Length];
                    for (int k = 0; k < x.Length / y.Length; k++)
                    {
                        sample[k] = x[j, k];
                    }

                    zVal[j] = y[j] - Infer(sample, i);
                }
                // Fit regression tree
                double[] yLoc = new double[y.Length];
                for (int j = 0; j < y.Length; j++)
                {
                    yLoc[j] = zVal[(int) regressionSources[testing][1][j]];
                }

                double[] newTree = Regression(regressionSources[testing][0], yLoc);
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

                
                lambdaNew = (1 + Math.Sqrt(1 + 4 * (Math.Pow(lambdaOld, 2)))) / 2;
                gamma = (1 - lambdaOld) / lambdaNew;
                lambdaOld = lambdaNew;
                _gamma_param[i + 1] = gamma;
            }

        }

        public static double[][][] RegressDataset(double[,] x, int dataLength)
        {
            double[][][] regressionSources = new double[x.Length / dataLength][][];
            for (int i = 0; i < x.Length / dataLength; i++)
            {
                regressionSources[i] = new double[2][];
                for (int j = 0; j < 2; j++)
                {
                    regressionSources[i][j] = new double[dataLength];
                    if (j == 1)
                    {
                        for (int k = 0; k < dataLength; k++)
                        {
                            regressionSources[i][j][k] = k;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < dataLength; k++)
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
            int[] result = new int[x.Length / _paramCount];
            for (int j = 0; j < x.Length / _paramCount; j++)
            {
                double[] sample = new double[_paramCount];
                for (int k = 0; k < _paramCount; k++)
                {
                    sample[k] = x[j, k];
                }

                result[j] = (int) Math.Round(Infer(sample, _epochs));
            }

            return result;
        }

        public double[] Regress(double[,] x)
        {
            double[] result = new double[x.Length / _paramCount];
            for (int j = 0; j < x.Length / _paramCount; j++)
            {
                double[] sample = new double[_paramCount];
                for (int k = 0; k < _paramCount; k++)
                {
                    sample[k] = x[j, k];
                }
                result[j] = Infer(sample, _epochs);
            }

            return result;
        }

        public double Infer(double[] sample, int inferBound, bool useG = false)
        {
            double ft = _model[0, 2];
            double gt = _model[0, 2];
            double ft1 = 0;
            double gt1 = 0;
            double treeReg;
            for (int i = 0; i < inferBound; i++)
            {
                if (sample[(int) _model[i + 1, 0]] < _model[i + 1, 1])
                {
                    treeReg = _model[i + 1, 2];
                }
                else
                {
                    treeReg = _model[i + 1, 3];
                }
                ft1 = gt + _shrinkage * treeReg;
                gt1 = (1 - _gamma_param[i + 1]) * ft1 + _gamma_param[i + 1] * ft;
                gt = gt1;
                ft = ft1;
            }

            if (!useG)
            {
                return ft1;
            }
            else
            {
                return gt1;
            }
        }
    }
}