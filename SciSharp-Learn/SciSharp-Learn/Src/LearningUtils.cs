using System.Linq;

namespace SciSharp_Learn
{
    public static class LearningUtils
    {
        public static double Accuracy(int[] predicted, int[] actual)
        {
            var correct = actual.Where((t, i) => predicted[i] == t).Count();

            return ((double) correct) / actual.Length;
        }
    }
}