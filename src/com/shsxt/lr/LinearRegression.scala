package com.shsxt.lr

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object LinearRegression {

    def main(args: Array[String]) {
        // 构建Spark对象
        val conf = new SparkConf().setAppName("LinearRegressionWithSGD").setMaster("local")
        val sc = new SparkContext(conf)
        Logger.getRootLogger.setLevel(Level.WARN)
        //    sc.setLogLevel("WARN")

        //读取样本数据
        val data_path1 = "lpsa.data"
        val data: RDD[String] = sc.textFile(data_path1)

        //1 为随机种子
        val train2TestData: Array[RDD[LabeledPoint]] = examples.randomSplit(Array(0.8, 0.2), 1)

        //    val numExamples = examples.count()

        // 迭代次数
        val numIterations = 100

        //在每次迭代的过程中 梯度下降算法的下降步长大小
        val stepSize = 0.1

        //每一次下山后，是否计算所有样本的误差值
        val miniBatchFraction = 1.0
        val lrs: LinearRegressionWithSGD = new LinearRegressionWithSGD()
        //设置需不需要有截距
//        lrs.setIntercept(true)
        lrs.optimizer.setStepSize(stepSize)
        lrs.optimizer.setNumIterations(numIterations)
        lrs.optimizer.setMiniBatchFraction(miniBatchFraction)

        val model: LinearRegressionModel = lrs.run(train2TestData(0))

        println(model.weights)
        println(model.intercept)

        // 对样本进行测试

        val prediction: RDD[Double] = model.predict(train2TestData(1).map(_.features))

        val predictionAndLabel: RDD[(Double, Double)] = prediction.zip(train2TestData(1).map(_.label))
        val print_predict: Array[(Double, Double)] = predictionAndLabel.take(100)
        println("prediction" + "\t" + "label")
        for (i <- 0 to print_predict.length - 1) {

            //            val tuple: (Double, Double) = print_predict(i)
            println(print_predict(i)._1 + "\t" + print_predict(i)._2)
        }
        // 计算测试误差
        val loss: Double = predictionAndLabel.map {
            case (p, v) =>
                val err = p - v
                Math.abs(err)
        }.reduce(_ + _)
        val error = loss / train2TestData(1).count
        println(s"Test RMSE = " + error)
//        // 模型保存
//        val ModelPath = "model"
////        model.save(sc, ModelPath)
////        val sameModel: LinearRegressionModel = LinearRegressionModel.load(sc, ModelPath)

        sc.stop()

    }

}
