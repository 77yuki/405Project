import os
import time
import luigi
import duckdb
import logging
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, split, when
import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1) Load & Transform eCommerce Data
class LoadECommerceData(luigi.Task):
    def output(self):
        return luigi.LocalTarget('intermediate/ecommerce_transformed')

    def run(self):
        logging.info('Starting eCommerce ETL Process')
        start_time = time.time()

        spark = SparkSession.builder.appName('ECommerce ETL').getOrCreate()
        try:
            logging.info('Loading eCommerce dataset...')
            ecom_df = spark.read.parquet('eCommerce_behavior_cleaned_for_etl_no_view.parquet').repartition(4)

            row_count = ecom_df.count()
            logging.info(f'Raw eCommerce row count: {row_count}')

            category_split = split(col('category_code'), '\\.')
            transformed_df = (
                ecom_df
                .withColumn('date', to_date(col('event_time'), 'yyyy-MM-dd'))
                .withColumn('is_purchase', when(col('event_type') == 'purchase', 1).otherwise(0))
                .withColumn('category_level1', category_split.getItem(0))
                .withColumn('category_level2', category_split.getItem(1))
                .withColumn('category_level3', category_split.getItem(2))
                .withColumn('category_level4', category_split.getItem(3))
            )

            transformed_count = transformed_df.count()
            logging.info(f'Transformed eCommerce row count: {transformed_count}')

            os.makedirs('intermediate', exist_ok=True)
            transformed_df.write.mode('overwrite').parquet(self.output().path)
            logging.info(f'Wrote Parquet to {self.output().path}')

            elapsed_time = time.time() - start_time
            logging.info(f'ECommerce task completed in {elapsed_time:.2f} seconds.')

        except Exception as e:
            logging.error(f'Error in eCommerce transformation: {e}')
            raise
        finally:
            spark.stop()


# 2) Load & Transform COVID Data
class LoadCovidData(luigi.Task):
    def output(self):
        return luigi.LocalTarget('intermediate/covid_transformed')

    def run(self):
        logging.info('Starting COVID ETL Process')

        spark = SparkSession.builder.appName('COVID ETL').getOrCreate()
        try:
            logging.info('Loading COVID dataset...')
            # Read CSV into Pandas, then create Spark DataFrame
            covid_pandas_df = pd.read_csv('cleaned_covid_data.csv')
            covid_df = spark.createDataFrame(covid_pandas_df).repartition(4)

            raw_count = covid_df.count()
            logging.info(f'Raw COVID row count: {raw_count}')

            # Convert date column from "M/d/yy" to Spark date, cast numeric columns
            # Then replace any NULL/NaN in total_cases/total_deaths with 0.0
            transformed_df = (
                covid_df
                .withColumn('date', to_date(col('date'), 'M/d/yy'))
                .withColumn('total_cases', col('total_cases').cast('double'))
                .withColumn('total_deaths', col('total_deaths').cast('double'))
                .withColumn('icu_patients', col('icu_patients').cast('double'))
                .withColumn('hosp_patients', col('hosp_patients').cast('double'))

                # For numeric columns that matter in aggregator, replace NaN/null with 0.0
                .withColumn('total_cases', 
                    F.when(F.isnan(col('total_cases')) | col('total_cases').isNull(), 0.0)
                     .otherwise(col('total_cases')))
                .withColumn('total_deaths', 
                    F.when(F.isnan(col('total_deaths')) | col('total_deaths').isNull(), 0.0)
                     .otherwise(col('total_deaths')))
            )

            transformed_count = transformed_df.count()
            logging.info(f'Transformed COVID row count: {transformed_count}')

            os.makedirs('intermediate', exist_ok=True)
            transformed_df.write.mode('overwrite').parquet(self.output().path)
            logging.info(f'Parquet file successfully written: {self.output().path}')

        except Exception as e:
            logging.error(f'Error in COVID transformation: {e}')
            raise
        finally:
            spark.stop()


# 3) Load Data into DuckDB and Create Views
class LoadToDuckDB(luigi.Task):
    def requires(self):
        return {
            'ecommerce': LoadECommerceData(),
            'covid': LoadCovidData()
        }

    def output(self):
        return luigi.LocalTarget('output/etl_complete.txt')

    def run(self):
        logging.info('Starting DuckDB Load Process')
        os.makedirs('output', exist_ok=True)

        ecommerce_path = 'intermediate/ecommerce_transformed'
        covid_path = 'intermediate/covid_transformed'

        if not os.path.exists(ecommerce_path) or not os.path.exists(covid_path):
            raise FileNotFoundError(f'Missing transformed Parquet at {ecommerce_path} or {covid_path}')

        con = duckdb.connect('output/retail_covid_analysis.db', read_only=False)
        try:
            con.execute('DROP TABLE IF EXISTS ecommerce')
            con.execute('DROP TABLE IF EXISTS covid')

            con.execute(f"CREATE TABLE ecommerce AS SELECT * FROM '{ecommerce_path}/*.parquet'")
            con.execute(f"CREATE TABLE covid AS SELECT * FROM '{covid_path}/*.parquet'")

            # Summarize COVID
            con.execute("""
                CREATE OR REPLACE VIEW covid_global AS
                SELECT 
                    date,
                    SUM(total_cases) AS global_total_cases,
                    SUM(total_deaths) AS global_total_deaths
                FROM covid
                GROUP BY date
            """)
            logging.info('Created aggregated covid_global view.')

            # Join on e.date = c.date
            con.execute("""
                CREATE OR REPLACE VIEW ecommerce_covid_joined AS
                SELECT
                    e.date,
                    e.event_type,
                    e.category_code,
                    e.category_level1,
                    e.category_level2,
                    e.category_level3,
                    e.category_level4,
                    e.brand,
                    e.price,
                    e.is_purchase,
                    c.global_total_cases,
                    c.global_total_deaths
                FROM ecommerce e
                LEFT JOIN covid_global c
                    ON e.date = c.date
                ORDER BY e.date
            """)
            logging.info('Created ecommerce_covid_joined view with global totals.')

            # Check a sample
            sample = con.execute("SELECT * FROM ecommerce_covid_joined LIMIT 5").fetchdf()
            logging.info(f"Sample from ecommerce_covid_joined:\n{sample}")

            # Mark success
            with open(self.output().path, 'w') as f:
                f.write('ETL completed successfully!\n')
            logging.info('DuckDB Load Completed.')

        except Exception as e:
            logging.error(f'Error in DuckDB Load: {e}')
            raise


# 4) Full ETL Pipeline
class ETLPipeline(luigi.Task):
    def requires(self):
        return LoadToDuckDB()

    def output(self):
        return luigi.LocalTarget('output/pipeline_complete.txt')

    def run(self):
        with open(self.output().path, 'w') as f:
            f.write('Pipeline completed successfully.\n')
        logging.info('ETL Pipeline Completed Successfully!')


if __name__ == '__main__':
    luigi.build([ETLPipeline()], local_scheduler=True, workers=1)
