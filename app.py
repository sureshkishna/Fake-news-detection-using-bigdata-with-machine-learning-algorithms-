from flask import Flask, render_template, request
from pyspark.ml.feature import Tokenizer, StopWordsRemover, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import StringIndexer
from xgboost.spark import SparkXGBClassifier

from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.functions import when, col, regexp_replace, concat, lit, length
from pyspark.sql.types import FloatType, DoubleType
# NLP
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession # to initiate spark
from pyspark.ml.feature import RegexTokenizer # tokenizer
from pyspark.ml.feature import HashingTF, IDF # vectorizer
from pyspark.ml.feature import StopWordsRemover # to remove stop words
from pyspark.sql.functions import concat_ws, col # to concatinate cols
from pyspark.ml.feature import VectorAssembler

# Initialize Flask app
app = Flask(__name__)

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("my_app_name") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Function to preprocess text
def preprocess_text(text):
    data = [(0, str(text)), (1, "x"), (2, "y")]

    # Create a DataFrame
    df = spark.createDataFrame(data, ["id", "text"])

    # Tokenize the text
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    wordsData = tokenizer.transform(df)

    # Compute Term Frequency (TF)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures" ,numFeatures=10000)
    tf = hashingTF.transform(wordsData)

    # Compute Inverse Document Frequency (IDF)
    idf = IDF(inputCol="rawFeatures", outputCol="feature")
    idfModel = idf.fit(tf)
    tfidf = idfModel.transform(tf)

    # Assemble the TF-IDF vectors into a single dense vector
    assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
    output = assembler.transform(tfidf)

    from pyspark.sql.functions import lit

    # Add a column with constant value 1 to each row
    output = output.withColumn("constant", lit(1))

    first_dense_vector_df = output.select("features").limit(1)

    # Show the result
    first_dense_vector_df.show(truncate=False)

    return first_dense_vector_df


def train_model():

    data_path = r"C:\Users\sures\Downloads\project\project\news.csv"
    spark_df = spark.read.csv(data_path, header=True, inferSchema=True)
    # Remove unimportant rows of the df

    spark_df = spark_df.filter((spark_df.label == 'FAKE') | (spark_df.label == 'REAL'))

    # Assuming 'label' is the name of the column containing the labels
    string_indexer = StringIndexer(inputCol='label', outputCol='encoded_label')
    spark_df = string_indexer.fit(spark_df).transform(spark_df)

    df_rmv_nan_text = spark_df.filter(length(col("text")) > 60)


    df_no_nan = (df_rmv_nan_text
                .withColumn("title", when(col("title") == "NaN", " ")
                                                .otherwise(col("title")))
                )


    df_clean = (df_no_nan

                    ## Removing any non-character from title
                    .withColumn("title",
                                regexp_replace(
                                    col('title'),
                                    r'[^\w\’ ]',''))

                    ## Removing any non-character from text
                    .withColumn("text",
                                regexp_replace(
                                    col('text'),
                                    r'[^\w\’ ]',''))

                    ## Replacing 2 or more whitespaces with 1 whitespace
                    .withColumn("text",
                                regexp_replace(
                                    col('text'),
                                    r'[ ]{2,}',' '))

                    ## Replacing 2 or more whitespaces with 1 whitespace
                    .withColumn("title",
                                regexp_replace(
                                    col('text'),
                                    r'[ ]{2,}',' '))
                    )

    df_combined = (df_clean
                        .withColumn('full_text',
                                    when(col("text").contains(
                                                        concat(col("title"))),
                                                        col("text"))

                                    .otherwise(concat(col("title"),
                                                        lit(" "),
                                                        col("text"))))
                        .select(["full_text","encoded_label"])
                        .withColumn("label", col("encoded_label").cast(DoubleType()))
                        .dropDuplicates()
                    )


    # Clean memory
    del df_rmv_nan_text, df_no_nan, df_clean

    try:
        stopwords_ls = stopwords.words('english')
    except:
        nltk.download("stopwords")
        stopwords_ls = stopwords.words('english')

    # Sanity Check
    stopwords_ls[:10]

    # Split data to train and test
    train, test = df_combined.randomSplit([0.7,0.3], seed=2)
    # convert sentences to list of words
    tokenizer = RegexTokenizer(inputCol="full_text", outputCol="words", pattern="\\W")

    train_df = tokenizer.transform(train)


    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filter")

    train_df= stopwords_remover.transform(train_df)


    # Calculate term frequency in each article
    hashing_tf = HashingTF(inputCol="filter", outputCol="raw_features", numFeatures=10000)
    featurized_data = hashing_tf.transform(train_df)

    # TF-IDF vectorization of articles
    idf = IDF(inputCol="raw_features", outputCol="features")
    idf_vectorizer = idf.fit(featurized_data)
    train_df = idf_vectorizer.transform(featurized_data)

    # Calculate term frequency in each article
    
    train=train_df.select("label","features")

    train =train.withColumn('label', train['label'].cast('double'))

    assembler = VectorAssembler(inputCols=['features'], outputCol='dense_features')
    train = assembler.transform(train)

    train.drop("features")

    from xgboost.spark import SparkXGBClassifier

    model=SparkXGBClassifier(label_col="label").fit(train)

    return model


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input text from the form
        input_text = request.form["news_text"]
        
        # Preprocess input text
        # Preprocess input text
        processed_df = preprocess_text(input_text)
        
        processed_features = processed_df.select("features").collect()[0]

        # Load trained model
        model = train_model()  # Assuming the model is trained and saved

        # Make prediction
        prediction = model.transform(spark.createDataFrame([processed_features], schema=["features"]))
        predicted_label = prediction.select("prediction").collect()[0][0]


        prediction.show()
        
        # Extract predicted label
        predicted_label = prediction.select("prediction").collect()[0][0]
        print("Predicted Label:", predicted_label)
        
        # Map label to human-readable form
        label_mapping = {0.0: "FAKE", 1.0: "REAL" }
        predicted_label_str = label_mapping[predicted_label]
        
        # Render prediction result template
        return render_template("result.html", input_text=input_text, prediction=predicted_label_str)
    else:
        # Render home page template
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
