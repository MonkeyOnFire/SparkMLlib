package com.shsxt.nb

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object Naive_bayes1 {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("word2vector").setMaster("local")
        val sc = new SparkContext(conf)
        val sqlContext = new SQLContext(sc)

        //加载数据
        val idData = sc.textFile("sms_spam.txt").map(_.split(",")).cache()
        //1.0为正常邮件 0.0为垃圾邮件
        val idDataRows: RDD[Row] = idData.map(x => Row((if (x(0) == "ham") 1.0 else 0.0), x(1).split(" ").map(_.trim)))


        val schema = StructType(List(
            StructField("label", DoubleType, nullable = false),
            StructField("words", ArrayType(StringType, true), nullable = false)
        ))

        val df = sqlContext.createDataFrame(idDataRows, schema)

        //构建词汇表/词袋 【i hate you love dont 】
        val countVectorizer: CountVectorizerModel =
            new CountVectorizer().setInputCol("words").setOutputCol("features").fit(df)

        //查看词汇表
        countVectorizer.vocabulary.take(100).foreach(println)


        //文本向量化 CountVector (1,2,1) IdfVector  WordVertor
        val cvDF: DataFrame = countVectorizer.transform(df)
        cvDF.show(false)

        //正负例样本，显示前10个
        val example: DataFrame = cvDF.drop("words")
        example.show(10)


        // 切分数据集与训练集
        val Array(trainingData, testData) = example.randomSplit(Array(0.8, 0.2), seed = 1234L)

        // 训练朴素贝叶斯模型
        val model: NaiveBayesModel = new NaiveBayes()
            .fit(trainingData)

        //
        // 预测  predict

        val predictions: DataFrame = model.transform(testData)


        predictions.show()
        //okmail: Dear Dave this is your final notice to collect your 4* Tenerife Holiday or #5000 CASH award!
        //模型评估
        predictions.registerTempTable("result")
        //计算正确率
        val accuracy: DataFrame = sqlContext.sql("select (1- (sum(abs(label-prediction)))/count(label)) as accuracy from result")
        accuracy.show()

        //保存模型
        //    model.save("sms_spam")

    }
}
