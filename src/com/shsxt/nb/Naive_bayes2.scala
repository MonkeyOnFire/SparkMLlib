package com.shsxt.nb

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
/**
  * 贝叶斯算法
  */
object Naive_bayes2 {
    def main(args: Array[String]) {
        //1 构建Spark对象
        val conf = new SparkConf().setAppName("Naive_bayes").setMaster("local")
        val sc = new SparkContext(conf)
        //读取样本数据1
        val data = sc.textFile("sample_naive_bayes_data.txt")
        val parsedData = data.map { line =>
            val parts = line.split(',')
            LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
        }

        //样本数据划分训练样本与测试样本
        val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0)
        val test = splits(1)

        //新建贝叶斯分类模型模型，并训练
        val model = NaiveBayes.train(training, lambda = 1.0)


        //对测试样本进行测试
        val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
        val print_predict = predictionAndLabel.take(20)
        println("prediction" + "\t" + "label")
        for (i <- 0 to print_predict.length - 1) {
            println(print_predict(i)._1 + "\t" + print_predict(i)._2)
        }

        val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
        println(accuracy)
        //保存模型
        val ModelPath = "naive_bayes_model"
        //    model.save(sc, ModelPath)
        //    val sameModel = NaiveBayesModel.load(sc, ModelPath)

    }
}
