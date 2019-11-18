using System;
using System.Collections.Generic;
using System.Linq;
using static SciSharp_Learn.InformationTheoryUtils;
using static SciSharp_Learn.LearningUtils;

namespace SciSharp_Learn
{
    public class DecisionTreeClassifier : IClassifier
    {
        private DecisionTree _tree;
        private int _attributeCount;
        private static int _discreteClasses;

        public DecisionTreeClassifier(int discreteClasses)
        {
            _discreteClasses = discreteClasses;
        }

        private static int MajorLabel(int[] y, int distinctLabelCount)
        {
            var labelCounts = new int[distinctLabelCount];
            foreach (var t in y)
            {
                labelCounts[t - y.Min()]++;
            }

            var maxLabelCount = 0;
            var chosenMaxLabel = -1;
            for (var i = 0; i < distinctLabelCount; i++)
            {
                if (maxLabelCount >= labelCounts[i]) continue;
                maxLabelCount = labelCounts[i];
                chosenMaxLabel = i;
            }

            return chosenMaxLabel;
        }

        private static IDecisionTreeNode IterativeDichotomiser3(int[,] x, int[] y, List<int> attributes)
        {
            var distinctLabelCount = y.Distinct().Count();
            if (x.Length == 0)
            {
                return new DecisionTreeLeafNode(0);
            }

            switch (distinctLabelCount)
            {
                case 1:
                    return new DecisionTreeLeafNode(y[0]);
                default:
                    {
                        if (attributes.Count == 0)
                        {
                            return new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                        }

                        var attributeTest = BestAttribute(x, y, attributes);
                        if (attributeTest == -1)
                        {
                            return new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                        }

                        attributes.Remove(attributeTest);
                        var attributeStateCount = 0;
                        for (var i = 0; i < y.Length; i++)
                        {
                            if (x[i, attributeTest] > attributeStateCount)
                            {
                                attributeStateCount = x[i, attributeTest];
                            }
                        }

                        ++attributeStateCount;
                        attributeStateCount = Math.Max(_discreteClasses * 2, attributeStateCount);

                        // Split data
                        var subX = new int[attributeStateCount][,];
                        var subY = new int[attributeStateCount][];
                        var attributeSubXCount = new int[attributeStateCount];
                        for (var i = 0; i < y.Length; i++)
                        {
                            ++attributeSubXCount[x[i, attributeTest]];
                        }

                        var attributeLength = x.Length / y.Length;
                        for (var i = 0; i < attributeStateCount; i++)
                        {
                            subX[i] = new int[attributeSubXCount[i], attributeLength];
                            subY[i] = new int[attributeSubXCount[i]];
                        }

                        var subXAttributeFormCount = new int[attributeStateCount];
                        for (var i = 0; i < y.Length; i++)
                        {
                            for (var j = 0; j < attributeLength; j++)
                            {
                                subX[x[i, attributeTest]][subXAttributeFormCount[x[i, attributeTest]], j] = x[i, j];
                            }

                            subY[x[i, attributeTest]][subXAttributeFormCount[x[i, attributeTest]]] = y[i];
                            ++subXAttributeFormCount[x[i, attributeTest]];
                        }

                        // Create tests
                        var tests = new IDecisionTreeNode[attributeStateCount];
                        for (var i = 0; i < attributeStateCount; i++)
                        {
                            if (subX[i].Length == 0)
                            {
                                tests[i] = new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                            }
                            else
                            {
                                tests[i] = IterativeDichotomiser3(subX[i], subY[i],
                                    attributes.GetRange(0, attributes.Count));
                            }
                        }

                        IDecisionTreeNode root = new DecisionTreeInternalNode(attributeTest, tests);

                        return root;
                    }
            }
        }

        public void Fit(double[,] x, int[] y)
        {
            _attributeCount = x.Length / y.Length;
            var newX = DiscreteFilter(x, _discreteClasses, _attributeCount);
            var attributes = new List<int>();
            for (var i = 0; i < _attributeCount; i++)
            {
                attributes.Add(i);
            }

            _tree = new DecisionTree(IterativeDichotomiser3(newX, y, attributes));
        }

        public int[] Predict(double[,] x)
        {
            var result = new int[x.Length / _attributeCount];
            var newX = DiscreteFilter(x, _discreteClasses, _attributeCount);
            for (var i = 0; i < x.Length / _attributeCount; i++)
            {
                var sample = new int[_attributeCount];
                for (var j = 0; j < _attributeCount; j++)
                {
                    sample[j] = newX[i, j];
                }

                result[i] = _tree.Classify(sample);
            }

            return result;
        }
    }
}