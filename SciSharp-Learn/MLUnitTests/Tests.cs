using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using static SciSharp_Learn.LinAlgUtils;
using static SciSharp_Learn.LearningUtils;
using static SciSharp_Learn.InformationTheoryUtils;
using static SciSharp_Learn.RegressionTreeStump;

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
            SgdClassifier model = new SgdClassifier(epochs: 10000, learningRate: 0.01);
            double[,] xTrain = new double[,] {{2, 0}, {1, 1}, {2, 3}, {6, 7}, {8, 9}, {1, 1}};
            int[] yTrain = new int[] {0, 0, 1, 1, 1, 0};
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            Console.WriteLine("Prediction:");
            foreach (var item in model.Predict(xTrain))
            {
                Console.WriteLine(item.ToString());
            }

            Console.WriteLine("Actual:");
            foreach (var item in yTrain)
            {
                Console.WriteLine(item.ToString());
            }

            Console.WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                Console.WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Assert.AreEqual(model.Predict(xTrain), yTrain);
        }

        [Test]
        public void TestAccuracyUtil()
        {
            var predicted = new[] {1, 1, 0, 0};
            var actual = new[] {1, 1, 1, 0};
            Assert.AreEqual(Accuracy(predicted, actual), 0.75);
        }

        [Test]
        public void DatasetBenchmarkSgdTest()
        {
            const string path =
                "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/heart.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, 14];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < 14; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
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

                yTrain[i] = (int) properties[i, 13];
            }

            var model = new SgdClassifier(epochs: 100000, learningRate: 0.01);
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            Console.WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            Console.WriteLine(accuracy);
            Console.WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                Console.WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Assert.Greater(accuracy, 0.75);
        }

        [Test]
        public void DatasetBenchmarkSgdTest2()
        {
            const string path =
                "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/pulsar_stars.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, 9];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < 9; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, 8];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < 8; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int) properties[i, 8];
            }

            var model = new SgdClassifier(epochs: 1000, learningRate: 0.01);
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            Console.WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            Console.WriteLine(accuracy);
            Console.WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                Console.WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Assert.Greater(accuracy, 0.75);
        }

        [Test]
        public void TestEntropy()
        {
            double[] probabilities = {1.0, 0.0};
            Assert.AreEqual(0, Entropy(probabilities));
            probabilities = new[] {0.5, 0.5};
            Assert.AreEqual(1, Entropy(probabilities));
            probabilities = new[] {0.25, 0.25, 0.25, 0.25};
            Assert.AreEqual(2, Entropy(probabilities));
            probabilities = new double[] {0, 1};
            Assert.AreEqual(0, Entropy(probabilities));
            probabilities = new[] {0.642857142857143D, 0.357142857142857D};
            Assert.AreEqual(0.940, Entropy(probabilities), 0.01);
        }

        [Test]
        public void TestProbabilities()
        {
            int[] y = {0, 0, 1, 0};
            double[] probabilities = {0.75, 0.25};
            Assert.AreEqual(probabilities, ProbabilityDistribution(y));
        }

        [Test]
        public void TestInformationGain()
        {
            int[,] x = new int[,] {{1, 1}, {1, 2}};
            int[] y = new int[] {0, 1};
            Assert.AreEqual(0, InformationGain(x, y, 0));
            Assert.AreEqual(1, InformationGain(x, y, 1));
        }

        [Test]
        public void TestBestAttribute()
        {
            int[,] x = {{1, 1}, {1, 2}};
            int[] y = {0, 1};
            var attr = new List<int> {0, 1};
            Assert.AreEqual(1, BestAttribute(x, y, attr));
        }

        [Test]
        public void TestDecisionTreeStructure()
        {
            IDecisionTreeNode first = new DecisionTreeLeafNode(2);
            DecisionTree tree = new DecisionTree(first);
            Assert.AreEqual(2, tree.Classify(new[] {1, 2, 3}));
            IDecisionTreeNode[] tests = new IDecisionTreeNode[]
            {
                new DecisionTreeLeafNode(0), new DecisionTreeLeafNode(0),
                new DecisionTreeLeafNode(1), new DecisionTreeLeafNode(2)
            };
            first = new DecisionTreeInternalNode(1, tests);
            tree = new DecisionTree(first);
            int[][] x = {new[] {0, 0}, new[] {0, 1}, new[] {0, 2}};
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(x[i][1], tree.Classify(x[i]));
            }
        }

        [Test]
        public void TestDecisionTreeClassification()
        {
            DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier(5);
            double[,] x = {{0, 0}, {0, 1}};
            int[] y = {0, 1};
            decisionTreeClassifier.Fit(x, y);
            int[] prediction = decisionTreeClassifier.Predict(new double[,] {{0, 0}, {0, 1}});
            Assert.AreEqual(new[] {0, 1}, prediction);
            x = new Double[,] {{0, 0}, {0, 1}, {0, 2}};
            y = new[] {0, 1, 2};
            decisionTreeClassifier.Fit(x, y);
            prediction = decisionTreeClassifier.Predict(new double[,] {{0, 0}, {0, 1}, {0, 2}});
            Assert.AreEqual(new[] {0, 1, 2}, prediction);
        }

        [Test]
        public void TestDiscreteFilter()
        {
            double[,] x = {{0, 0.12}, {0, 234}};
            int[,] filtered = DiscreteFilter(x, 2, 2);
            Assert.AreEqual(new[,] {{0, 0}, {0, 1}}, filtered);
        }

        [Test]
        public void DatasetBenchmarkDtTest()
        {
            int featuresCount = 8;
            const string path =
                "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/pulsar_stars.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, featuresCount + 1];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < featuresCount + 1; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, featuresCount];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < featuresCount; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int) properties[i, featuresCount];
            }

            var model = new DecisionTreeClassifier(5);
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            Console.WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            Console.WriteLine(accuracy);
            Assert.Greater(accuracy, 0.65);
        }

        [Test]
        public void DatasetBenchmarkDtTest2()
        {
            int featuresCount = 13;
            const string path =
                "/home/karol_oleszek/Projects/SciSharpLearn/SciSharp-Learn/SciSharp-Learn/MLUnitTests/BenchmarkDatasets/heart.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, featuresCount + 1];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < featuresCount + 1; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, featuresCount];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < featuresCount; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int) properties[i, featuresCount];
            }

            var model = new DecisionTreeClassifier(20);
            model.Fit(xTrain, yTrain);
            Console.WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            Console.WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            Console.WriteLine(accuracy);
            Assert.Greater(accuracy, 0.65);
        }

        [Test]
        public void TestRegressionTreeStump()
        {
            double[] x = new double[] {0, 1, 2, 3};
            double[] y = new double[] {1, 1, 2, 2};
            Assert.AreEqual(new double[] {1.5, 1, 2}, Regression(x, y));
        }
    }
}