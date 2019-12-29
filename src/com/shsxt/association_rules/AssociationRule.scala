package com.shsxt.association_rules

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer


object AssociationRule {

    /**
      * Spark购物篮关联规则算法
      **/
    def main(args: Array[String]): Unit = {
        val inputPath = "shopping_cart"
        val outputPath = "rs/shopping_cart"

        val sparkConf: SparkConf = new SparkConf()
            .setMaster("local")
            .setAppName("AssociationRule")
        val sc: SparkContext = SparkContext.getOrCreate(sparkConf)
        val transactions: RDD[String] = sc.textFile(inputPath)
        //求出商品组合：(List(a,b),1) (List(b,c),1) (List(a,b,c),1)...
        val patterns: RDD[(List[String], Int)] = transactions.flatMap(line => {
            val items = line.split(",").toList

            (0 to items.size).flatMap(items.combinations).filter(xs => !xs.isEmpty)
        }).map((_, 1))
        
        //商品组合出现的频度计算
        /**
          *
          * (List(b, c, e),1)
          * (List(b, d, e),1)
          * (List(c, d),2)
          * (List(c, e),2)
          * (List(b),9)
          * (List(a, b, d),1)
          * (List(b, d),5)
          * (List(a, b),3)
          * (List(d, e),2)
          *
          */
        val combined: RDD[(List[String], Int)] = patterns.reduceByKey(_ + _)

        combined.collect().foreach(println)

        /**
          * 算出所有的关联规则
          * 
          * (List(b, c, e),(List(),1))
          * (List(c, e),(List(b, c, e),1))
          * (List(b, e),(List(b, c, e),1))
          * (List(b, c),(List(b, c, e),1))
          * (List(b, d, e),(List(),1))
          * (List(e),(List(c, e),2))
          * (List(c),(List(c, e),2))
          *
          */
        val subpatterns: RDD[(List[String], (List[String], Int))] = combined.flatMap(pattern => {
            val result = ListBuffer.empty[Tuple2[List[String], Tuple2[List[String], Int]]]
            result += ((pattern._1, (Nil, pattern._2)))
            print(result)
            val sublist= for {
                i <- 0 until pattern._1.size
                xs = pattern._1.take(i) ++ pattern._1.drop(i + 1)
                if xs.size > 0
            } yield (xs, (pattern._1, pattern._2))
            result ++= sublist
            println(" : " + result.toList)
            result.toList
        })

        subpatterns.collect().foreach(x => {println(x + "-----------")})

        val rules: RDD[(List[String], Iterable[(List[String], Int)])] = subpatterns.groupByKey()

        //计算每个规则的概率
        val assocRules: RDD[List[(List[String], List[String], Double)]] = rules.map(in => {
//            val a: Iterable[(List[String], Int)] = in._2
            val fromCount = in._2.find(p => p._1 == Nil).get
            val lstData = in._2.filter(p => p._1 != Nil).toList
            if (lstData.isEmpty) Nil
            else {
                val result = {
                    for {
                        t2 <- lstData
                        confidence = t2._2.toDouble / fromCount._2.toDouble
                        difference = t2._1 diff in._1
                    } yield (((in._1, difference, confidence)))
                }
                result
            }
        })

        val formatResult: RDD[(String, String, Double)] = assocRules.flatMap(f => {
            f.map(s => (s._1.mkString("[", ",", "]"), s._2.mkString("[", ",", "]"), s._3))
        }).sortBy(tuple => tuple._3, false, 1)
        //保存结果
        //formatResult.saveAsTextFile(outputPath)

        //打印商品组合频度
        combined.foreach(println)
        //打印商品关联规则和置信度
        formatResult.foreach(println)
        sc.stop()
    }

}