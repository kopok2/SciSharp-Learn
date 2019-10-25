namespace SciSharp_Learn
{
    public interface IClassifier
    {
        void Fit(double[,] x, int[] y);
        int[] Predict(double[,] x);
        double Score();
    }
}