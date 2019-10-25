namespace SciSharp_Learn
{
    public interface IClassifier
    {
        void Fit();
        int[] Predict();
        double Score();
    }
}