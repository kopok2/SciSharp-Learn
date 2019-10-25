using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using SciSharp_Learn;

namespace MLUnitTests
{
    [TestFixture]
    public class Tests
    {
        [Test]
        public void TestDotMatrixVector()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[] vector = new double[] {7, 12};
            double[] expectedResult = new double[] {12, 50};
            Assert.AreEqual(SgdClassifier.DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            vector = new double[] {1, 2, 3};
            expectedResult = new double[] {1, 2, 3};
            Assert.AreEqual(SgdClassifier.DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] {{3}};
            vector = new double[] {5};
            expectedResult = new double[] {15};
            Assert.AreEqual(SgdClassifier.DotMatrixVector(matrix, vector), expectedResult);
        }

        [Test]
        public void TestSigmoid()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[] beta = new double[] {7, 12};
            double[] expectedResult = new double[] {0.9999938558253978, 1.0};
            Assert.AreEqual(SgdClassifier.Sigmoid(matrix, beta), expectedResult);
        }

        [Test]
        public void TestMatrixTranspose()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[,] expectedResult = new double[,] {{0, 2}, {1, 3}};
            Assert.AreEqual(SgdClassifier.MatrixTranspose(matrix, 2), expectedResult);
            matrix = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            expectedResult = new double[,] {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
            Assert.AreEqual(SgdClassifier.MatrixTranspose(matrix, 3), expectedResult);
        }

        [Test]
        public void TestLogisticGradient()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[] beta = new double[] {7, 12};
            double[] target = new double[] {1, 1};
            double[] expectedResult = new double[] {0, 0.9999938558253978 - 1};
            Assert.AreEqual(SgdClassifier.LogisticGradient(matrix, beta, target), expectedResult);
        }

        [Test]
        public void TestMeanSquaredError()
        {
            double[] y1 = new double[] {1, 2, 3};
            double[] y2 = new double[] {1, 3, 10};
            double expectedResult = 50.0;
            Assert.AreEqual(SgdClassifier.MeanSquaredError(y1, y2), expectedResult);
        }

        [Test]
        public void TestSGD()
        {
            SgdClassifier model = new SgdClassifier(aepochs:10000, alearningRate:0.01);
            double[,] x_train = new double[,] {{2,0},{1,1},{2,3},{6,7},{8, 9},{1,1}};
            int[] y_train = new int[] {0, 0, 1, 1, 1, 0};
            model.Fit(x_train, y_train);
            System.Console.WriteLine("Training data:");
            System.Console.WriteLine("Prediction:");
            foreach(var item in model.Predict(x_train))
            {
                Console.WriteLine(item.ToString());
            }
            System.Console.WriteLine("Actual:");
            foreach(var item in y_train)
            {
                Console.WriteLine(item.ToString());
            }
            System.Console.WriteLine("Beta params:");
            foreach(var item in model.betaParam)
            {
                Console.WriteLine(item.ToString());
            }
            Assert.AreEqual(model.Predict(x_train), y_train);
        }

        [Test]
        public void DatasetBenchmarkSGDTest()
        {
            string path =
                "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/heart.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            double[,] properties = new Double[lineCount,14];
            for(int i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for(int i = 0; i < 14; i++)
                {
                    
                    if(line!=null)
                    {
                        var values = line.Split(',');
                        properties[i2,i] = Convert.ToDouble(values[i]);
                    }
                }
            }
            double[,] x_train = new double[lineCount, 13];
            int[] y_train = new int[lineCount];
            for (int i = 0; i < lineCount; i++)
            {
                for (int j = 0; j < 13; j++)
                {
                    x_train[i, j] = properties[i, j];
                }

                y_train[i] = (int)properties[i, 13];
            }
            SgdClassifier model = new SgdClassifier(aepochs:10000, alearningRate:0.01);
            model.Fit(x_train, y_train);
            System.Console.WriteLine("Training data:");
            System.Console.WriteLine("Prediction:");
            foreach(var item in model.Predict(x_train))
            {
                Console.WriteLine(item.ToString());
            }
            System.Console.WriteLine("Actual:");
            foreach(var item in y_train)
            {
                Console.WriteLine(item.ToString());
            }

            int corr = 0;
            int incorr = 0;
            for (int i = 0; i < y_train.Length; i++)
            {
                
            }
            System.Console.WriteLine("Beta params:");
            foreach(var item in model.betaParam)
            {
                Console.WriteLine(item.ToString());
            }
            Assert.AreEqual(model.Predict(x_train), y_train);
        }
    }
}