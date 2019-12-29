package com.shsxt.tfidf


import java.io.StringReader

import org.apache.spark.{SparkContext, SparkConf}
import org.wltea.analyzer.core.{Lexeme, IKSegmenter}

import scala.collection.mutable.ArrayBuffer

object TFIDF {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local").setAppName("TFIDF")
    val sc = new SparkContext(conf)

    val corpus = sc.textFile("original.txt").map(_.split("\t")(1))
    corpus.map({ str =>
      val reader = new StringReader(str)
      val ik: IKSegmenter = new IKSegmenter(reader, true)
      var term: Lexeme = null
      val arr = new ArrayBuffer[String]()
      do {
        term = ik.next()
      }
      while (term != null)
    {
      arr.append(term.getLexemeText)
      term = ik.next()
    }
      arr.toArray
    }).foreach(println(_))
  }
}
