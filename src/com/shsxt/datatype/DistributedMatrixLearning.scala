/**
  * @author root
  *         ref:Spark MlLib机器学习实战
  *         more code:https://github.com/root245/SparkLearning
  *         more blog:http://blog.csdn.net/root245
  */
package com.shsxt.datatype

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object DistributedMatrixLearning {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("DistributedMatrix")
    val sc = new SparkContext(conf)

    println("行矩阵 ")
    val rdd = sc.textFile("data/MatrixRow.txt") //创建RDD文件路径
      .map(_.split(' ') //按“ ”分割
      .map(_.toDouble)) //转成Double类型
      .map(line => Vectors.dense(line)) //转成Vector格式
    val rm = new RowMatrix(rdd) //读入行矩阵

    //打印行矩阵所有值
    println(rm.numRows()) //打印列数
    println(rm.numCols()) //打印行数
    rm.rows.foreach(println)

    val summary = rm.computeColumnSummaryStatistics()
    // 可以通过summary实例来获取矩阵的相关统计信息，例如行数
    println(summary.count)
    // 最大向量
    println(summary.max)
    // 方差向量
    println(summary.variance)
    // 平均向量
    println(summary.mean)
    // L1范数向量
    println(summary.normL1)
    

    //索引矩阵
    println("索引矩阵 ")
    val rdd2 = sc.textFile("data/MatrixRow.txt") //创建RDD文件路径
      .map(_.split(' ') //按“ ”分割
      .map(_.toDouble)) //转成Double类型
      .map(line => Vectors.dense(line)) //转化成向量存储
      .map((vd) => new IndexedRow(vd.size, vd)) //转化格式
    val irm = new IndexedRowMatrix(rdd2) //建立索引行矩阵实例
    println(irm.getClass) //打印类型
    irm.rows.foreach(println) //打印内容数据

    println("坐标矩阵 ")
    val rdd3 = sc.textFile("data/MatrixRow.txt") //创建RDD文件路径
      .map(_.split(' ') //按“ ”分割
      .map(_.toDouble)) //转成Double类型
//      .map(vue => (vue(0).toLong, vue(1).toLong, vue(2))) //转化成坐标格式
//      .map(vue2 => new MatrixEntry(vue2 _1, vue2 _2, vue2 _3)) //转化成坐标矩阵格式
      .map(vue2 => new MatrixEntry(vue2(0).toLong, vue2(1).toLong, vue2(2))) //转化成坐标矩阵格式

    val crm = new CoordinateMatrix(rdd3) //实例化坐标矩阵
    //转置
    val crmT = crm.transpose()

    crm.entries.foreach(println) //打印数据
    crmT.entries.foreach(println) //打印数据
    println(crm.numCols())
    println(crm.numRows())

    println("分块矩阵")
    /*
        1.0   0.0  0.0  0.0
        0.0   1.0  0.0  0.0
        -1.0  2.0  1.0  0.0
        1.0   1.0  0.0  1.0
    */
    import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
    // 创建8个矩阵项，每一个矩阵项都是由索引和值构成的三元组
    val ent1 = new MatrixEntry(0,0,1)
    val ent2 = new MatrixEntry(1,1,1)
    val ent3 = new MatrixEntry(2,0,-1)
    val ent4 = new MatrixEntry(2,1,2)
    val ent5 = new MatrixEntry(2,2,1)
    val ent6 = new MatrixEntry(3,0,1)
    val ent7 = new MatrixEntry(3,1,1)
    val ent8 = new MatrixEntry(3,3,1)

    // 创建RDD[MatrixEntry]
    val entries : RDD[MatrixEntry] = sc.parallelize(Array(ent1,ent2,ent3,ent4,ent5,ent6,ent7,ent8))

    // 通过RDD[MatrixEntry]创建一个坐标矩阵
    val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)

    // 将坐标矩阵转换成2x2的分块矩阵并存储，尺寸通过参数传入
    val matA: BlockMatrix = coordMat.toBlockMatrix(2,2).cache()
      // 可以用validate()方法判断是否分块成功
      matA.validate()
    //构建成功后，可通过toLocalMatrix转换成本地矩阵，并查看其分块情况：
    println(matA.toLocalMatrix)
    matA.blocks.foreach(println)
    sc.stop
  }
}
