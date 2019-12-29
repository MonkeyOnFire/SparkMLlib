package com.shsxt.datatype

import org.apache.spark.mllib.linalg.{Matrix, Matrices}

object MatrixTest {

  def main(args: Array[String]) {



    //稠密矩阵
    val denseMatrix = Matrices.dense(3,2,Array(1.0,2.0,1.0,3.0,2.0,1.0))
    println(denseMatrix)

    //稀疏矩阵
    val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))
    println(sm)

    /*
    1,1,0,3
    2,3,0,0
    1,2,0,1
    */

    val sm1 = Matrices.sparse(3,4,Array(0,3,6,6,8),Array(0,1,2,0,1,2,0,2),Array(1,2,1,1,3,2,3,1))
    println(sm1)

  }

}
