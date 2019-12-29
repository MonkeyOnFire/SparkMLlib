package com.shsxt.kmeans

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
object KMeans2 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KMeans2").setMaster("local")
    val sc = new SparkContext(conf)

    val rdd = sc.parallelize(List(
      Vectors.dense(Array(-0.1, 0.0, 0.0)),
      Vectors.dense(Array(9.0, 9.0, 9.0)),
      Vectors.dense(Array(3.0, 2.0, 1.0))))

    val centroids = sc.textFile("kmeans_data.txt")
        .map(_.split(" ").map(_.toDouble))
        .map(Vectors.dense(_))
        .collect()
//    val centroids = Array(
//      Vectors.dense(Array(0.0, 0.0, 0.0)),
//      Vectors.dense(Array(0.1, 0.1, 0.1)),
//      Vectors.dense(Array(0.2, 0.2, 0.2)),
//      Vectors.dense(Array(9.0, 9.0, 9.0)),
//      Vectors.dense(Array(9.1, 9.1, 9.1)),
//      Vectors.dense(Array(9.2, 9.2, 9.2)))

    val model = new KMeansModel(clusterCenters=centroids)
    
    model.clusterCenters.foreach { println }
    
    model.predict(rdd).collect().foreach(println(_))

  }
}
