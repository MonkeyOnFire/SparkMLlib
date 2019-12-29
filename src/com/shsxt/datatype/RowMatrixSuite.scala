/**
  * @author root
  *         ref:http://spark.apache.org/docs/1.5.2/mllib-guide.html
  *         more code:https://github.com/root245/SparkLearning
  *         more blog:http://blog.csdn.net/root245
  */
package com.shsxt.datatype

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
object RowMatrixSuite {

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("RowMatrixSuite")

    //    println("start sc")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    //    val rdd = sc.parallelize(Array(1, 2, 3))
    //    println("count:" + rdd.count())

    val rdd = sc.textFile("MatrixRow33.txt") //创建RDD文件路径
      .map(_.split(' ') //按“ ”分割
      .map(_.toDouble)) //转成Double类型
      .map(line => Vectors.dense(line)) //转成Vector格式
    val mat = new RowMatrix(rdd)

    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()
    println("m:" + m)
    println("n:" + n)
    println("mat:" + mat)

    sc.stop()
  }
}
