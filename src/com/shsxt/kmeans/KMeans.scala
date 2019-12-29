package com.shsxt.kmeans

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object KMeans {

    def main(args: Array[String]) {
        //1 构建Spark对象
        val conf = new SparkConf().setAppName("KMeans").setMaster("local")
        val sc = new SparkContext(conf)

        // 读取样本数据1，格式为LIBSVM format
        val data = sc.textFile("kmeans_data.txt")
        val parsedData: RDD[linalg.Vector] = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()


        val numClusters = 4
        val numIterations = 100
        val model: KMeansModel = new KMeans().
            setK(numClusters).
            setMaxIterations(numIterations).
            run(parsedData)
        val centers: Array[linalg.Vector] = model.clusterCenters
        println("centers")
        for (i <- 0 to centers.length - 1) {
            println(centers(i)(0) + "\t" + centers(i)(1) + "\t" + centers(i)(2))
        }

        // 误差计算
        val WSSSE = model.computeCost(parsedData)

        println("Errors = " + WSSSE)

        println(model.predict(Vectors.dense(10, 10, 10)))
        //保存模型
        //    val ModelPath = "KMeans_Model"
        //    model.save(sc, ModelPath)
        //    val sameModel = KMeansModel.load(sc, ModelPath)
    }
}
