package com.shsxt.lr

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegression6 {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("spark").setMaster("local[3]")
        val sc = new SparkContext(conf)
        val inputData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "环境分类数据.txt")

        val vectors: RDD[linalg.Vector] = inputData.map(_.features)
        val scalerModel: StandardScalerModel = new StandardScaler(withMean = true, withStd = true).fit(vectors)
        val normalizeInputData: RDD[LabeledPoint] = inputData.map { point =>
            val label = point.label
            val features = scalerModel.transform(point.features.toDense)
            println(features + "=====================")
            new LabeledPoint(label, features)
        }

        val splits = normalizeInputData.randomSplit(Array(0.7, 0.3))

        //        val splits = inputData.randomSplit(Array(0.7, 0.3),1L)
        val (trainingData, testData) = (splits(0), splits(1))
        val lr = new LogisticRegressionWithLBFGS()

        lr.setIntercept(true)
        val model: LogisticRegressionModel = lr.run(trainingData)
        val result = testData
            .map { point => Math.abs(point.label - model.predict(point.features)) }
        println("正确率=" + (1.0 - result.mean()))
        println(model.weights.toArray.mkString(" "))
        println(model.intercept)

    }
}
