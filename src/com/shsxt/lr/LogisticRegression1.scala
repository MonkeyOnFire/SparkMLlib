package com.shsxt.lr

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegression1 {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("spark").setMaster("local[3]")
        val sc = new SparkContext(conf)
        val inputData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "健康状况训练集.txt")
        val splits: Array[RDD[LabeledPoint]] = inputData.randomSplit(Array(0.7, 0.3), seed = 1L)
        val (trainingData, testData) = (splits(0), splits(1))
        val lr = new LogisticRegressionWithLBFGS()
        lr.setIntercept(true)
        val model = lr.run(trainingData)
        //0 1:18 2:0 3:2 4:3 5:2 6:5
        val result: RDD[Double] = testData
            .map { point => Math.abs(point.label - model.predict(point.features)) }
        println("正确率=" + (1.0 - result.mean()))
        println(model.weights.toArray.mkString(" "))
        println(model.intercept)
    }
}