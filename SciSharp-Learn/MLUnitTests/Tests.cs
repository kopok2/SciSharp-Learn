using System;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using static SciSharp_Learn.LinAlgUtils;

namespace SciSharp_Learn
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
            Assert.AreEqual(DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            vector = new double[] {1, 2, 3};
            expectedResult = new double[] {1, 2, 3};
            Assert.AreEqual(DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] {{3}};
            vector = new double[] {5};
            expectedResult = new double[] {15};
            Assert.AreEqual(DotMatrixVector(matrix, vector), expectedResult);
        }

        [Test]
        public void TestSigmoid()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[] beta = new double[] {7, 12};
            double[] expectedResult = new double[] {0.9999938558253978, 1.0};
            Assert.AreEqual(Sigmoid(matrix, beta), expectedResult);
        }

        [Test]
        public void TestMatrixTranspose()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[,] expectedResult = new double[,] {{0, 2}, {1, 3}};
            Assert.AreEqual(MatrixTranspose(matrix, 2), expectedResult);
            matrix = new double[,] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
            expectedResult = new double[,] {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
            Assert.AreEqual(MatrixTranspose(matrix, 3), expectedResult);
        }

        [Test]
        public void TestLogisticGradient()
        {
            double[,] matrix = new double[,] {{0, 1}, {2, 3}};
            double[] beta = new double[] {7, 12};
            double[] target = new double[] {1, 1};
            double[] expectedResult = new double[] {0, 0.9999938558253978 - 1};
            Assert.AreEqual(LogisticGradient(matrix, beta, target), expectedResult);
        }

        [Test]
        public void TestMeanSquaredError()
        {
            double[] y1 = new double[] {1, 2, 3};
            double[] y2 = new double[] {1, 3, 10};
            double expectedResult = 50.0;
            Assert.AreEqual(MeanSquaredError(y1, y2), expectedResult);
        }

        [Test]
        public void TestSgd()
        {
            SgdClassifier model = new SgdClassifier(epochs:10000, learningRate:0.01);
            double[,] xTrain = new double[,] {{2,0},{1,1},{2,3},{6,7},{8, 9},{1,1}};
            int[] yTrain = new int[] {0, 0, 1, 1, 1, 0};
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            Console.WriteLine("Prediction:");
            foreach(var item in model.Predict(xTrain))
            {
                Console.WriteLine(item.ToString());
            }
            Console.WriteLine("Actual:");
            foreach(var item in yTrain)
            {
                Console.WriteLine(item.ToString());
            }
            Console.WriteLine("Beta params:");
            foreach(var item in model.BetaParam)
            {
                Console.WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }
            Assert.AreEqual(model.Predict(xTrain), yTrain);
        }

        [Test]
        public void DatasetBenchmarkSgdTest()
        {
            const string path = "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/heart.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount,14];
            for(var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for(var i = 0; i < 14; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2,i] = Convert.ToDouble(values[i]);
                }
            }
            var xTrain = new double[lineCount, 13];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < 13; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int)properties[i, 13];
            }
            var model = new SgdClassifier(epochs:100000, learningRate:0.01);
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            double correct = 0;
            double incorrect = 0;
            var predicted = model.Predict(xTrain);
            for (var i = 0; i < yTrain.Length; i++)
            {
                if (predicted[i] == yTrain[i])
                {
                    ++correct;
                }
                else
                {
                    ++incorrect;
                }
            }
            Console.WriteLine("Accuracy:");
            Console.WriteLine(correct / (correct + incorrect));
            Console.WriteLine("Beta params:");
            foreach(var item in model.BetaParam)
            {
                Console.WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }
            Assert.Greater(correct / (correct + incorrect), 0.75);
        }
    }
}