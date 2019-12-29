package com.shsxt.datatype

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object DataTypeTest {


    def main(args: Array[String]): Unit = {
//    val conf = new SparkConf().setMaster("local").setAppName("DataTypeTest")
//    val sc = new SparkContext(conf)

    //稠密向量
    val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
    //稀疏向量
    val sv1: Vector = Vectors.sparse(10, Array(0, 2,5), Array(1.0, 3.0,8.0))
//    val sv2: Vector = Vectors.sparse(10, Seq((0, 1.0), (2, 3.0)))

    println(dv)
    println(sv1)
//    println(sv2)

    println(sv1.toDense)
//
    val pos: LabeledPoint = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
    val neg: LabeledPoint = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
    println(pos.label)
    println(neg.features.toDense)
//
//    val labeledPointRDD = sc.parallelize(Array(pos,neg));
//    //存储标注点数据
//    MLUtils.saveAsLibSVMFile(labeledPointRDD,"labeledPointRDD.txt")
//    //加载标注点数据
//    val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
//
//    examples.foreach { x =>
//      {
//        val label = x.label
//        val features = x.features
//        println("label:" + label + "\tfeatures:" + features.toDense)
//      }
//    }

//    sc.stop()

  }
}