package com.shsxt.wv

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
object Word2Vector {
    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName("word2vector").setMaster("local")
        val sc = new SparkContext(conf)
        val idData: RDD[Seq[String]] = sc.textFile("doc").map(_.split("[^a-zA-Z]").toSeq).map(_.map(_.toLowerCase)).cache()
        val word2vec = new Word2Vec().setVectorSize(2)// 近义词 同义词

        val model: Word2VecModel = word2vec.fit(idData)
        //like(0.0475,0.4328) love(0.0485,0.4028)
        println(model.transform("only"))
        val wordVectors: Map[String, Array[Float]] = model.getVectors
        wordVectors.foreach { case (word, vector) => println(word + "\t的词向量是\t" + vector.toSeq) }
        model.findSynonyms("spark", 30).foreach(x => println(x._1))
    }
}
