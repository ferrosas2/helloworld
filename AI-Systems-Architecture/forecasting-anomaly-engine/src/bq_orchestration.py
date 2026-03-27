import logging
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BigQueryMLPipeline:
    def __init__(self, project_id=None):
        try:
            self.client = bigquery.Client(project=project_id)
            logging.info(f"Initialized BigQuery Client for project: {self.client.project}")
        except GoogleCloudError as e:
            logging.error(f"Failed to initialize BigQuery Client: {e}")
            raise

    def _read_sql_file(self, filepath):
        try:
            with open(filepath, 'r') as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"SQL script file not found: {filepath}")
            raise

    def _execute_query(self, query):
        try:
            job = self.client.query(query)
            logging.info("Executing BigQuery Job...")
            result = job.result()
            logging.info(f"Job finished. Bytes processed: {job.total_bytes_processed}")
            return result
        except Exception as e:
            logging.error(f"Error executing BigQuery Job: {e}")
            raise

    def run_training_job(self, sql_path):
        logging.info("Starting run_training_job")
        query_job = self._read_sql_file(sql_path)
        self._execute_query(query_job)
        logging.info("Training job completed successfully.")

    def get_anomalies(self, sql_path):
        logging.info("Starting get_anomalies")
        query_job = self._read_sql_file(sql_path)
        result = self._execute_query(query_job)
        return result.to_dataframe()

    def get_explainability(self, sql_path):
        logging.info("Starting get_explainability")
        query_job = self._read_sql_file(sql_path)
        result = self._execute_query(query_job)
        return result.to_dataframe()
