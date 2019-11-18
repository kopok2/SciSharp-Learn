using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using static NUnit.Framework.Assert;
using static SciSharp_Learn.InformationTheoryUtils;
using static SciSharp_Learn.LearningUtils;
using static SciSharp_Learn.LinAlgUtils;
using static SciSharp_Learn.RegressionTreeStump;
using static System.Console;

namespace SciSharp_Learn
{
    
    [TestFixture]
    public class Tests
    {
        private const string BasePath = "C:\\Users\\Mateusz\\Desktop\\SciSharp-Learnnn\\SciSharp-Learn\\MLUnitTests\\BenchmarkDatasets";
        [Test]
        public void TestDotMatrixVector()
        {
            var matrix = new double[,] { { 0, 1 }, { 2, 3 } };
            var vector = new double[] { 7, 12 };
            var expectedResult = new double[] { 12, 50 };
            AreEqual(DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            vector = new double[] { 1, 2, 3 };
            expectedResult = new double[] { 1, 2, 3 };
            AreEqual(DotMatrixVector(matrix, vector), expectedResult);
            matrix = new double[,] { { 3 } };
            vector = new double[] { 5 };
            expectedResult = new double[] { 15 };
            AreEqual(DotMatrixVector(matrix, vector), expectedResult);
        }
        [Test]
        public void TestKnnMockData()
        {
            double[,] stringdata = new Double[6, 2] { { 1, 1 }, { 1, 3 }, { 2, 1 }, { 3, 4 }, { 4, 5 }, { 5, 5 } };
            int[] prediction = new int[6] { 1, 1, 1, 0, 0, 0 };
            double[,] predict = new double[2, 2] { { 0, 0 }, { 4, 5 } };

            Knn Model = new Knn();
            Model.Fit(stringdata, prediction);
            int[] predicted;
            predicted = Model.Predict(predict);
            int[] ShoudBe = new int[predicted.Length];
            ShoudBe[0] = 1;
            ShoudBe[1] = 0;
            for (int i = 0; i < predicted.GetLength(0); i++)
            {
                AreEqual(predicted[i], ShoudBe[i]);
            }
        }
        [Test]
        public void TestNeuralNetworkMockData()
        {
            double[,] stringdata = new Double[6, 2] { { 1, 1 }, { 1, 3 }, { 2, 1 }, { 3, 4 }, { 4, 5 }, { 5, 5 } };
            int[] prediction = new int[6] { 1, 1, 1, 0, 0, 0 };
            int[] predicted;
            double[,] predict = new double[2, 2] { { 0, 0 }, { 5, 5 } };
            NeuralNetworkClassifier NN = new NeuralNetworkClassifier();
            NN.Fit(stringdata, prediction);
            Console.WriteLine("prediction made");
            predicted = NN.Predict(predict);

            int[] ShoudBe = new int[predicted.Length];
            ShoudBe[0] = 1;
            ShoudBe[1] = 0;
            for (int i = 0; i < predicted.GetLength(0); i++)
            {
                AreEqual(predicted[i], ShoudBe[i]);
            }
        }



        [Test]
        public void TestSigmoid()
        {
            var matrix = new double[,] { { 0, 1 }, { 2, 3 } };
            var beta = new double[] { 7, 12 };
            var expectedResult = new[] { 0.9999938558253978, 1.0 };
            AreEqual(Sigmoid(matrix, beta), expectedResult);
        }

        [Test]
        public void TestMatrixTranspose()
        {
            var matrix = new double[,] { { 0, 1 }, { 2, 3 } };
            var expectedResult = new double[,] { { 0, 2 }, { 1, 3 } };
            AreEqual(MatrixTranspose(matrix, 2), expectedResult);
            matrix = new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            expectedResult = new double[,] { { 1, 4, 7 }, { 2, 5, 8 }, { 3, 6, 9 } };
            AreEqual(MatrixTranspose(matrix, 3), expectedResult);
        }

        [Test]
        public void TestLogisticGradient()
        {
            var matrix = new double[,] { { 0, 1 }, { 2, 3 } };
            var beta = new double[] { 7, 12 };
            var target = new double[] { 1, 1 };
            var expectedResult = new[] { 0, 0.9999938558253978 - 1 };
            AreEqual(LogisticGradient(matrix, beta, target), expectedResult);
        }

        [Test]
        public void TestMeanSquaredError()
        {
            var y1 = new double[] { 1, 2, 3 };
            var y2 = new double[] { 1, 3, 10 };
            const double expectedResult = 50.0;
            AreEqual(MeanSquaredError(y1, y2), expectedResult);
        }

        [Test]
        public void TestSgd()
        {
            var model = new SgdClassifier(epochs: 10000, learningRate: 0.01);
            var xTrain = new double[,] { { 2, 0 }, { 1, 1 }, { 2, 3 }, { 6, 7 }, { 8, 9 }, { 1, 1 } };
            var yTrain = new[] { 0, 0, 1, 1, 1, 0 };
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            WriteLine("Prediction:");
            foreach (var item in model.Predict(xTrain))
            {
                WriteLine(item.ToString());
            }

            WriteLine("Actual:");
            foreach (var item in yTrain)
            {
                WriteLine(item.ToString());
            }

            WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            AreEqual(model.Predict(xTrain), yTrain);
        }

        [Test]
        public void TestAccuracyUtil()
        {
            var predicted = new[] { 1, 1, 0, 0 };
            var actual = new[] { 1, 1, 1, 0 };
            AreEqual(Accuracy(predicted, actual), 0.75);
        }

        [Test]
        public void DatasetBenchmarkSgdTest()
        {
            const string path = BasePath + "//heart.csv";
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

                yTrain[i] = (int)properties[i, 13];
            }

            var model = new SgdClassifier(epochs: 100000, learningRate: 0.01);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Greater(accuracy, 0.75);
        }

        [Test]
        public void DatasetBenchmarkSgdTest2()
        {
            const string path = BasePath+
                "//pulsar_stars.csv";
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

                yTrain[i] = (int)properties[i, 8];
            }

            var model = new SgdClassifier(epochs: 1000, learningRate: 0.01);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Greater(accuracy, 0.75);
        }

        [Test]
        public void DatasetBenchmarkSgdTest3()
        {
            const string path = BasePath+
                "\\ldp.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, 22];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < 22; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, 21];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < 21; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int)properties[i, 21];
            }

            var model = new SgdClassifier(epochs: 10000, learningRate: 0.005);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            WriteLine("Beta params:");
            foreach (var item in model.BetaParam)
            {
                WriteLine(item.ToString(CultureInfo.InvariantCulture));
            }

            Greater(accuracy, 0.75);
        }

        [Test]
        public void TestEntropy()
        {
            double[] probabilities = { 1.0, 0.0 };
            AreEqual(0, Entropy(probabilities));

            probabilities = new[] { 0.5, 0.5 };
            AreEqual(1, Entropy(probabilities));

            probabilities = new[] { 0.25, 0.25, 0.25, 0.25 };
            AreEqual(2, Entropy(probabilities));

            probabilities = new double[] { 0, 1 };
            AreEqual(0, Entropy(probabilities));

            probabilities = new[] { 0.642857142857143D, 0.357142857142857D };
            AreEqual(0.940, Entropy(probabilities), 0.01);
        }

        [Test]
        public void TestProbabilities()
        {
            int[] y = { 0, 0, 1, 0 };
            double[] probabilities = { 0.75, 0.25 };
            AreEqual(probabilities, ProbabilityDistribution(y));
        }

        [Test]
        public void TestInformationGain()
        {
            var x = new[,] { { 1, 1 }, { 1, 2 } };
            var y = new[] { 0, 1 };
            AreEqual(0, InformationGain(x, y, 0));
            AreEqual(1, InformationGain(x, y, 1));
        }

        [Test]
        public void TestBestAttribute()
        {
            int[,] x = { { 1, 1 }, { 1, 2 } };
            int[] y = { 0, 1 };
            var attr = new List<int> { 0, 1 };
            AreEqual(1, BestAttribute(x, y, attr));
        }

        [Test]
        public void TestDecisionTreeStructure()
        {
            IDecisionTreeNode first = new DecisionTreeLeafNode(2);
            var tree = new DecisionTree(first);
            AreEqual(2, tree.Classify(new[] { 1, 2, 3 }));
            var tests = new IDecisionTreeNode[]
            {
                new DecisionTreeLeafNode(0), new DecisionTreeLeafNode(0),
                new DecisionTreeLeafNode(1), new DecisionTreeLeafNode(2)
            };
            first = new DecisionTreeInternalNode(1, tests);
            tree = new DecisionTree(first);
            int[][] x = { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 0, 2 } };
            for (var i = 0; i < 3; i++)
            {
                AreEqual(x[i][1], tree.Classify(x[i]));
            }
        }

        [Test]
        public void TestDecisionTreeClassification()
        {
            var decisionTreeClassifier = new DecisionTreeClassifier(5);
            double[,] x = { { 0, 0 }, { 0, 1 } };
            int[] y = { 0, 1 };
            decisionTreeClassifier.Fit(x, y);
            var prediction = decisionTreeClassifier.Predict(new double[,] { { 0, 0 }, { 0, 1 } });
            AreEqual(new[] { 0, 1 }, prediction);
            x = new double[,] { { 0, 0 }, { 0, 1 }, { 0, 2 } };
            y = new[] { 0, 1, 2 };
            decisionTreeClassifier.Fit(x, y);
            prediction = decisionTreeClassifier.Predict(new double[,] { { 0, 0 }, { 0, 1 }, { 0, 2 } });
            AreEqual(new[] { 0, 1, 2 }, prediction);
        }

        [Test]
        public void TestDiscreteFilter()
        {
            double[,] x = { { 0, 0.12 }, { 0, 234 } };
            var filtered = DiscreteFilter(x, 2, 2);
            AreEqual(new[,] { { 0, 0 }, { 0, 1 } }, filtered);
        }

        [Test]
        public void DatasetBenchmarkDtTest()
        {
            const int featuresCount = 8;
            const string path = BasePath + 
                "\\pulsar_stars.csv";
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

                yTrain[i] = (int)properties[i, featuresCount];
            }

            var model = new DecisionTreeClassifier(5);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            Greater(accuracy, 0.65);
        }

        [Test]
        public void DatasetBenchmarkDtTest2()
        {
            int featuresCount = 13;
            const string path = BasePath + 
                "\\heart.csv";
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

                yTrain[i] = (int)properties[i, featuresCount];
            }

            var model = new DecisionTreeClassifier(20);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            double accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            Greater(accuracy, 0.65);
        }

        [Test]
        public void DatasetBenchmarkDtTest3()
        {
            const string path = BasePath +
                "\\ldp.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, 22];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < 22; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, 21];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < 21; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int)properties[i, 21];
            }

            var model = new DecisionTreeClassifier(20);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);

            Greater(accuracy, 0.75);
        }

        [Test]
        public void TestRegressionTreeStump()
        {
            double[] x = { 0, 1, 2, 3 };
            double[] y = { 1, 1, 2, 2 };
            AreEqual(new[] { 1.5, 1, 2 }, Regression(x, y));
        }

        [Test]
        public void TestDatasetRegressionPreprocess()
        {
            double[,] x = { { 0, 1 }, { 2, 3 }, { 0, 4 } };
            AreEqual(
                new[]
                {
                    new[] {new double[] {0, 0, 2}, new double[] {0, 2, 1}},
                    new[] {new double[] {1, 3, 4}, new double[] {0, 1, 2}}
                }, AcceleratedGradientBoostingClassifier.RegressDataset(x, 3));
        }

        [Test]
        public void TestAcceleratedGradientBoostingClassifier()
        {
            AcceleratedGradientBoostingClassifier agbc = new AcceleratedGradientBoostingClassifier(50, 0.8);
            double[,] x = { { 0, 0, 0 }, { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } };
            int[] y = { 1, 2, 3, 4 };
            agbc.Fit(x, y);
            PrintDataset(agbc.Predict(x));
            PrintDataset(agbc.Regress(x));
            var prediction = agbc.Predict(x);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(prediction, y);
            WriteLine(accuracy);
            AreEqual(accuracy, 1.0);
        }

        [Test]
        public void 
            DatasetBenchmarkAgbcTest()
        {
            const string path = BasePath +
                "\\heart.csv";
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

                yTrain[i] = (int)properties[i, 13];
            }

            var model = new AcceleratedGradientBoostingClassifier(epochs: 100, shrinkage: 0.9);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            PrintDataset(yTrain);
            PrintDataset(predicted);
            Greater(accuracy, 0.65);
        }

        [Test]
        public void DatasetBenchmarkAgbcTest2()
        {
            const string path = BasePath +
                "\\pulsar_stars.csv";
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

                yTrain[i] = (int)properties[i, 8];
            }

            var model = new AcceleratedGradientBoostingClassifier(epochs: 100, shrinkage: 0.9);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);

            Greater(accuracy, 0.75);
        }
        [Test]
        public void DatasetBenchmarkAgbcTest3()
        {
            const string path = BasePath +
                "\\ldp.csv";
            var lineCount = File.ReadLines(path).Count();
            var reader = new StreamReader(File.OpenRead(path));
            var properties = new double[lineCount, 22];
            for (var i2 = 0; i2 < lineCount; i2++)
            {
                var line = reader.ReadLine();
                for (var i = 0; i < 22; i++)
                {
                    if (line == null) continue;
                    var values = line.Split(',');
                    properties[i2, i] = Convert.ToDouble(values[i]);
                }
            }

            var xTrain = new double[lineCount, 21];
            var yTrain = new int[lineCount];
            for (var i = 0; i < lineCount; i++)
            {
                for (var j = 0; j < 21; j++)
                {
                    xTrain[i, j] = properties[i, j];
                }

                yTrain[i] = (int)properties[i, 21];
            }

            var model = new AcceleratedGradientBoostingClassifier(epochs: 100, shrinkage: 0.9);
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);

            Greater(accuracy, 0.75);
        }
        [Test]
        public void DatasetBenchmarkNeuralNetwork()
        {
            int featuresCount = 8;
            const string path = BasePath + 
                "\\heart.csv";
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

                yTrain[i] = (int)properties[i, featuresCount];
            }
            var model = new NeuralNetworkClassifier();
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);



            Greater(accuracy, 0.60);
        }
        [Test]
        public void DatasetBenchmarkKNN()
        {
            int featuresCount = 8;
            const string path = BasePath +
                "\\heart.csv";
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

                yTrain[i] = (int)properties[i, featuresCount];
            }
            var model = new Knn();
            model.Fit(xTrain, yTrain);
            WriteLine("Training data:");
            var predicted = model.Predict(xTrain);
            WriteLine("Accuracy:");
            var accuracy = Accuracy(predicted, yTrain);
            WriteLine(accuracy);
            Greater(accuracy, 0.70);
        }
    }
}