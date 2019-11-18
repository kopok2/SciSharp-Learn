using System;

namespace SciSharp_Learn
{
    public class NeuralNetworkClassifier : IClassifier
    {
        public NeuralNetwork NN { get; set; }
        public void Fit(double[,] x, int[] y)
        {
            double[][] AllData = new double[x.GetLength(0)][];
            Random rnd = new Random();
            int seed = rnd.Next(1, 10);
            int RowNumber = x.GetLength(0);
            int FeatureNumber = x.GetLength(1);//equal to num input
            int OutputSize = 2;// Change to get distinc valeus of y[]
            for (int i = 0; i < x.GetLength(0); i++)
            {
                double[] vec = new double[x.GetLength(1) + OutputSize];
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    vec[j] = x[i, j];
                }
                //static typing for bainary clasification
                if (y[i] == 1)
                {
                    vec[x.GetLength(1)] = 1;
                    vec[x.GetLength(1) + 1] = 0;
                }
                else
                {
                    vec[x.GetLength(1)] = 0;
                    vec[x.GetLength(1) + 1] = 1;
                }
                AllData[i] = vec;
            }

            double[][] trainData;

            double[][] testData;
            SplitTrainTest(AllData, 1, seed, out trainData, out testData);
            ShowMatrix(trainData, 4, 2, true);
            NN = new NeuralNetwork(FeatureNumber, FeatureNumber * 2, OutputSize);
            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;

            double[] weights = NN.Train(trainData, maxEpochs, learnRate, momentum);

            double trainAcc = NN.Accuracy(trainData);
            Console.WriteLine("\nFinal accuracy on training data = " +
                              trainAcc.ToString("F4"));
        }
        public int[] Predict(double[,] x)
        {
            int[] prediction = new int[x.GetLength(0)];
            double[][] AllData = new double[x.GetLength(0)][];
            for (int i = 0; i < x.GetLength(0); i++)
            {
                double[] vec = new double[x.GetLength(1)];
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    vec[j] = x[i, j];
                }
                AllData[i] = NN.ComputeOutputs(vec);
                if (AllData[i][0] > 0.8)
                {
                    prediction[i] = 1;
                }
                else
                {
                    prediction[i] = 0;
                }
            }
            return prediction;
        }
        static void SplitTrainTest(double[][] allData, double trainPct,
            int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // SplitTrainTest

        public static void ShowMatrix(double[][] matrix, int numRows,
            int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }
    }
    public class NeuralNetwork
    {

        private int numInput; // number input nodes
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights; // input-hidden
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights; // hidden-output
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            this.rnd = new Random(0);
            this.InitializeWeights(); // all weights and biases
        } 

        private static double[][] MakeMatrix(int rows,
          int cols, double v) // helper for ctor, Train
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }
        private void InitializeWeights() // helper for ctor
        {
            // initialize weights and biases to small random values
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) +
              (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j]; 

            for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)   // apply activation
                this.hOutputs[i] = HyperTan(hSums[i]);

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to output sums
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs 
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0; 
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / sum;

            return result;
        }

        public double[] Train(double[][] trainData, int maxEpochs,
          double learnRate, double momentum)
        {
            // train using back-prop
            // back-prop specific arrays
            double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
            double[] obGrads = new double[numOutput];                   // output bias gradients

            double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
            double[] hbGrads = new double[numHidden];                   // hidden bias gradients

            double[] oSignals = new double[numOutput];                  // local gradient output signals - gradients w/o associated input terms
            double[] hSignals = new double[numHidden];                  // local gradient hidden node signals

            // back-prop momentum specific arrays 
            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput];

            int epoch = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // target values
            double derivative = 0.0;
            double errorSignal = 0.0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 10; // interval to check error
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    double trainErr = Error(trainData);
                    Console.WriteLine("epoch = " + epoch + "  error = " +
                      trainErr.ToString("F4"));
                    //Console.ReadLine();
                }

                Shuffle(sequence); // visit each training data in random order
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs 

                    // indices: i = inputs, j = hiddens, k = outputs

                    // 1. compute output node signals (assumes softmax)
                    for (int k = 0; k < numOutput; ++k)
                    {
                        errorSignal = tValues[k] - outputs[k];  
                        derivative = (1 - outputs[k]) * outputs[k]; 
                        oSignals[k] = errorSignal * derivative;
                    }

                    // 2. compute hidden-to-output weight gradients using output signals
                    for (int j = 0; j < numHidden; ++j)
                        for (int k = 0; k < numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * hOutputs[j];

                    // 2b. compute output bias gradients using output signals
                    for (int k = 0; k < numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

                    // 3. compute hidden node signals
                    for (int j = 0; j < numHidden; ++j)
                    {
                        derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
                        double sum = 0.0; // need sums of output signals times hidden-to-output weights
                        for (int k = 0; k < numOutput; ++k)
                        {
                            sum += oSignals[k] * hoWeights[j][k]; // represents error signal
                        }
                        hSignals[j] = derivative * sum;
                    }

                    // 4. compute input-hidden weight gradients
                    for (int i = 0; i < numInput; ++i)
                        for (int j = 0; j < numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    // 4b. compute hidden node bias gradients
                    for (int j = 0; j < numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

                    // == update weights and biases

                    // update input-to-hidden weights
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double delta = ihGrads[i][j] * learnRate;
                            ihWeights[i][j] += delta; // would be -= if (o-t)
                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta; // save for next time
                        }
                    }

                    // update hidden biases
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // update hidden-to-output weights
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int k = 0; k < numOutput; ++k)
                        {
                            double delta = hoGrads[j][k] * learnRate;
                            hoWeights[j][k] += delta;
                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // update output node biases
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // while
            double[] bestWts = GetWeights();
            return bestWts;
        } // Train

        private void Shuffle(int[] sequence) // instance method
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        } // Shuffle

        private double Error(double[][] trainData)
        {
            // average squared error per training item
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        } // MeanSquaredError

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // get x-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }

    } // NeuralNet
}
