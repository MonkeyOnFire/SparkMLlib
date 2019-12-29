package com.shsxt.lr

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.optimization.{L1Updater, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegression5 {

    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName("spark").setMaster("local[3]")
        val sc = new SparkContext(conf)
        val inputData = MLUtils.loadLibSVMFile(sc, "健康状况训练集.txt")
        val splits = inputData.randomSplit(Array(0.8, 0.2))
        val (trainingData, testData) = (splits(0), splits(1))
       // val lr = new LogisticRegressionWithLBFGS() // 拟牛顿法 ,没有L1正则化，只有L2
        val lr = new LogisticRegressionWithSGD() // 随机梯度下降算法，两个都有
        //
        lr.setIntercept(true)
        lr.optimizer.setUpdater(new L1Updater)
        //    LogisticRegressionWithSGD 才有L1正则优化
//        lr.optimizer.setUpdater(new SquaredL2Updater)

        // 这块设置的是我们的lambda,越大越看重这个模型的推广能力,一般不会超过1,0.4是个比较好的值
        lr.optimizer.setRegParam(0.4)
        val model = lr.run(trainingData)
        val result = testData
            .map { point => Math.abs(point.label - model.predict(point.features)) }
        println("正确率=" + (1.0 - result.mean()))
        println(model.weights.toArray.mkString(" "))
        println(model.intercept)

        /**
          * 正确率=0.736613972463029
0.02243946134462616 0.08469067285039861 -0.017819054685646454 -0.08018179788353921 -0.07244424603666609 -0.08337982547139504
-0.0342302016202474
          */

    }
}
