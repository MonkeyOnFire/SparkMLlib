package com.shsxt.kmeans

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by zfg on 2017/8/14.
  */
object KMeans1 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KMeans1").setMaster("local")
    val sc = new SparkContext(conf)
    val data = sc.textFile("kmeans_data.txt")
    val parsedData = data.map(line => Vectors.dense(line.split(' ').map(_.toDouble)))

    val numClusters = 3
    val numIterations = 5
    val model = org.apache.spark.mllib.clustering.KMeans.train(parsedData, numClusters, numIterations)
    model.clusterCenters.foreach(println(_))

//    对已经存在的点进行判断
    println(model.predict(Vectors.dense("0.2 0.2 0.2".split(' ').map(_.toDouble))))
//    对不存在的点进行判断
    println(model.predict(Vectors.dense("8 8 8".split(' ').map(_.toDouble))))

//    评估平方误差
    val WSSSE = model.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
  }
}
