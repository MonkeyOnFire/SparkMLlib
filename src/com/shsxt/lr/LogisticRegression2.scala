package com.shsxt.lr

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object LogisticRegression2 {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("spark").setMaster("local[3]")
        val sc = new SparkContext(conf)
        val inputData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "w0测试数据.txt")
        val splits = inputData.randomSplit(Array(0.7, 0.3))
        val (trainingData, testData) = (splits(0), splits(1))
        val lr = new LogisticRegressionWithSGD

        // 设置要有W0
        lr.setIntercept(true)
        val model: LogisticRegressionModel = lr.run(trainingData)
        val result = testData
            .map { point => Math.abs(point.label - model.predict(point.features)) }
        println("正确率=" + (1.0 - result.mean()))
        println(model.weights.toArray.mkString(" "))
        println(model.intercept)


    }
}
