using System;
using System.Collections.Generic;
using System.Linq;


namespace SciSharp_Learn
{ public class Knn : IClassifier
    {
        public double[,] BaseArray { get; set; }

        public void Fit(double[,] x, int[] y)
        {
            BaseArray = new double[x.GetLength(0), x.GetLength(1) + 1];
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    BaseArray[i, j] = x[i, j];
                }

                for (int j = x.GetLength(1); j < x.GetLength(1) + 1; j++)
                {
                    BaseArray[i, j] = (double)y[i];
                }
            }


        }
        public int[] Predict(double[,] x)
        {
            int K;//make k pulic 
            K = 3;
            int[] result = new int[x.GetLength(0)];
            int RowNumber;
            RowNumber = x.GetLength(0);
            int ColumnNumber = x.GetLength(1);

            for (int i = 0; i < RowNumber; i++)
            {
                double[] Distance = new double[BaseArray.GetLength(0)];
                int[] Label = new int[BaseArray.GetLength(0)];
                for (int k = 0; k < BaseArray.GetLength(0); k++)
                {


                    for (int j = 0; j < ColumnNumber; j++)
                    {
                        Distance[k] += CalulateDistance(x[i, j], BaseArray[k, j]);



                    }

                    Label[k] = (int)BaseArray[k, ColumnNumber];


                }

                Array.Sort(keys: Distance, items: Label);

                var LabelCut = Label.Take(K);

                result[i] = LabelCut.GroupBy(ii => ii).OrderBy(g => g.Count()).Last().Key;


            }

            return result;
        }
        public double CalulateDistance(double value1, double Value2)
        {
            return Math.Pow(value1 - Value2, 2);
        }

    }
    class Sort
    {
        public static double[,] SortByComlun(double[,] arr, int column)
        {
            double[][] arrJaged;
            arrJaged = ToJagged(arr);
            WriteRows(arr, arrJaged);
            SortJ(arrJaged, column);
            arr = ToRectangular(arrJaged);
            return arr;
        }
        public static void ShowMatrix<T>(T[,] arr)
        {
            int rowLength = arr.GetLength(0);
            int colLength = arr.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", arr[i, j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }
        public static void ShowMatrix<T>(T[][] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                System.Console.Write("Element({0}): ", i);

                for (int j = 0; j < arr[i].Length; j++)
                {
                    System.Console.Write("{0}{1}", arr[i][j], j == (arr[i].Length - 1) ? "" : " ");
                }
                System.Console.WriteLine();
            }
        }

        private static void SortJ<T>(T[][] data, int col)
        {
            Comparer<T> comparer = Comparer<T>.Default;
            Array.Sort<T[]>(data, (x, y) => comparer.Compare(x[col], y[col]));
        }
        public static double[][] ToJagged(double[,] array)
        {
            int height = array.GetLength(0), width = array.GetLength(1);
            double[][] jagged = new Double[height][];

            for (int i = 0; i < height; i++)
            {
                double[] row = new Double[width];
                for (int j = 0; j < width; j++)
                {
                    row[j] = array[i, j];
                }
                jagged[i] = row;
            }
            return jagged;
        }
        public static double[,] ToRectangular(double[][] array)
        {
            int height = array.Length, width = array[0].Length;
            double[,] rect = new double[height, width];
            for (int i = 0; i < height; i++)
            {
                double[] row = array[i];
                for (int j = 0; j < width; j++)
                {
                    rect[i, j] = row[j];
                }
            }
            return rect;
        }
        public static void WriteRows(double[,] array, params double[][] rows)
        {
            for (int i = 0; i < rows.Length; i++)
            {
                double[] row = rows[i];
                for (int j = 0; j < row.Length; j++)
                {
                    array[i, j] = row[j];
                }
            }
        }
    }
}

