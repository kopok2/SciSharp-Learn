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
        private readonly int _discreteClasses;

        public DecisionTreeClassifier(int discreteClasses)
        {
            _discreteClasses = discreteClasses;
        }

        private static int MajorLabel(int[] y, int distinctLabelCount)
        {
            int[] labelCounts = new int[distinctLabelCount];
            for (int i = 0; i < y.Length; i++)
            {
                labelCounts[y[i] - y.Min()]++;
            }

            int maxLabelCount = 0;
            int chosenMaxLabel = -1;
            for (int i = 0; i < distinctLabelCount; i++)
            {
                if (maxLabelCount < labelCounts[i])
                {
                    maxLabelCount = labelCounts[i];
                    chosenMaxLabel = i;
                }
            }

            return chosenMaxLabel;
        }
        private static IDecisionTreeNode IterativeDichotomiser3(int[,] x, int[] y, List<int> attributes)
        {
            int distinctLabelCount = y.Distinct().Count();
            if (distinctLabelCount == 1)
            {
                return new DecisionTreeLeafNode(y[0]);
            }
            else
            {
                if (attributes.Count == 0)
                {
                    return new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                }
                else
                {
                    int attributeTest = BestAttribute(x, y);
                    if (attributeTest == -1)
                    {
                        return new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                    }
                    attributes.Remove(attributeTest);
                    int attributeStateCount = 0;
                    for (int i = 0; i < y.Length; i++)
                    {
                        if (x[i, attributeTest] > attributeStateCount)
                        {
                            attributeStateCount = x[i, attributeTest];
                        }
                    }

                    ++attributeStateCount;

                    // Split data
                    int[][,]subX = new int[attributeStateCount][,];
                    int[][]subY = new int[attributeStateCount][];
                    int[] attributeSubXCount = new int[attributeStateCount];
                    for (int i = 0; i < y.Length; i++)
                    {
                        ++attributeSubXCount[x[i, attributeTest]];
                    }

                    int attributeLength = x.Length / y.Length;
                    for (int i = 0; i < attributeStateCount; i++)
                    {
                        subX[i] = new int[attributeSubXCount[i], attributeLength] ;
                        subY[i] = new int[attributeSubXCount[i]];
                    }
                    int[] subXAttributeFormCount = new int[attributeStateCount];
                    for (int i = 0; i < y.Length; i++)
                    {
                        for (int j = 0; j < attributeLength; j++)
                        {
                            subX[x[i, attributeTest]][subXAttributeFormCount[x[i, attributeTest]], j] = x[i, j];
                        }

                        subY[x[i, attributeTest]][subXAttributeFormCount[x[i, attributeTest]]] = y[i];
                        ++subXAttributeFormCount[x[i, attributeTest]];
                    }
                    
                    // Create tests
                    IDecisionTreeNode[] tests = new IDecisionTreeNode[attributeStateCount];
                    for (int i = 0; i < attributeStateCount; i++)
                    {
                        if (subX[i].Length == 0)
                        {
                            tests[i] = new DecisionTreeLeafNode(MajorLabel(y, distinctLabelCount));
                        }
                        else
                        {
                            tests[i] = IterativeDichotomiser3(subX[i], subY[i], attributes);
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
            int[,] newX = DiscreteFilter(x, _discreteClasses, _attributeCount);
            var attributes = new List<int>();
            for (int i = 0; i < _attributeCount; i++)
            {
                attributes.Add(i);
            }
            _tree = new DecisionTree(IterativeDichotomiser3(newX, y, attributes));
        }

        public int[] Predict(double[,] x)
        {
            int[]result = new int[x.Length / _attributeCount];
            int[,] newX = DiscreteFilter(x, _discreteClasses, _attributeCount);
            for (int i = 0; i < x.Length / _attributeCount; i++)
            {
                int[]sample = new int[_attributeCount];
                for (int j = 0; j < _attributeCount; j++)
                {
                    sample[j] = newX[i, j];
                }

                result[i] = _tree.Classify(sample);
            }

            return result;
        }
    }
}