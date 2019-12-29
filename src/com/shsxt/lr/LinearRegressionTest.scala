package com.shsxt.lr

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 线性回归,房价预测
  */
object LinearRegressionTest {

    def main(args: Array[String]) {
        // 构建Spark对象
        val conf = new SparkConf().setAppName("LinearRegressionWithSGD").setMaster("local")
        val sc = new SparkContext(conf)
        Logger.getRootLogger.setLevel(Level.WARN)
        //    sc.setLogLevel("WARN")

        //读取样本数据
        val data_path1 = "house_data.csv"
        val data = sc.textFile(data_path1)
        val examples = data.map { line =>
            val parts = line.split(',')
            val tmp = parts(5) :: parts(12) :: Nil
            LabeledPoint(parts(13).toDouble, Vectors.dense(tmp.toArray.map(_.toDouble)))
//            LabeledPoint(parts(13).toDouble, Vectors.dense(parts.slice(0, 13).map(_.toDouble)))
        }.cache()

        examples.foreach(println)


        val train2TestData = examples.randomSplit(Array(0.8, 0.2), 1)

        val numExamples = examples.count()

        // 迭代次数
        val numIterations = 50

        //在每次迭代的过程中 梯度下降算法的下降步长大小
        //        val stepSize = 1
        val stepSize = 0.01

        //每一次下山后，是否计算所有样本的误差值
        val miniBatchFraction = 1
        val lrs = new LinearRegressionWithSGD()

        lrs.setIntercept(true)
        lrs.optimizer.setStepSize(stepSize)
        lrs.optimizer.setNumIterations(numIterations)
        lrs.optimizer.setMiniBatchFraction(miniBatchFraction)
        lrs.optimizer.setConvergenceTol(0.0001)
        val model = lrs.run(train2TestData(0))
        println(model.weights)
        println(model.intercept)
        /**
          *
          * 对样本进行测试;x.features为测试数据的特征值
          * prediction 为： 测试结果集即预测值(y值)
          *
          */

        val prediction: RDD[Double] = model.predict(train2TestData(1).map(x => {
            x.features
        }))
        val predictionAndLabel = prediction.zip(train2TestData(1).map(_.label))
        val print_predict = predictionAndLabel.take(20)
        println("prediction" + "\t" + "label")
        for (i <- 0 to print_predict.length - 1) {
            println(print_predict(i)._1 + "\t" + print_predict(i)._2)
        }
        // 计算测试误差
        val loss = predictionAndLabel.map {
            case (p, v) =>
                val err = p - v
                Math.abs(err)
        }.reduce(_ + _)
        val error = loss / train2TestData(1).count
        println(s"Test RMSE = " + error)
        //模型保存
        val ModelPath = "model"
        //        model.save(sc, ModelPath)
        //        val sameModel = LinearRegressionModel.load(sc, ModelPath)

        sc.stop()
    }

}
