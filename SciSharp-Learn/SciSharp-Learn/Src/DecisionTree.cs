namespace SciSharp_Learn
{
    public class DecisionTreeInternalNode : IDecisionTreeNode
    {
        private readonly int _testingAttribute;
        private readonly IDecisionTreeNode[] _testNodes;

        public DecisionTreeInternalNode(int testingAttribute, IDecisionTreeNode[] testNodes)
        {
            _testingAttribute = testingAttribute;
            _testNodes = testNodes;
        }

        public int Decide(int[] x)
        {
            return _testNodes[x[_testingAttribute] + 1].Decide(x);
        }
    }

    public class DecisionTreeLeafNode : IDecisionTreeNode
    {
        private readonly int _leafDecision;

        public DecisionTreeLeafNode(int leafDecision)
        {
            _leafDecision = leafDecision;
        }

        public int Decide(int[] x)
        {
            return _leafDecision;
        }
    }

    public class DecisionTree
    {
        private readonly IDecisionTreeNode _root;

        public DecisionTree(IDecisionTreeNode root)
        {
            _root = root;
        }

        public int Classify(int[] x)
        {
            return _root.Decide(x);
        }
    }
}