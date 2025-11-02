from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
import tempfile
import os
import csv

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='s3_csv_to_postgres_copy',
    default_args=default_args,
    description='Load CSV from S3 to PostgreSQL with auto table creation',
    schedule='@daily',
    catchup=False,
    tags=['etl', 's3', 'postgres', 'copy'],
) as dag:

    @task
    def create_table_from_csv(s3_bucket: str, s3_key: str, table_name: str,
                              aws_conn_id: str = 'aws_default',
                              postgres_conn_id: str = 'postgres_default'):
        """Inspect CSV and create table schema automatically"""
        
        s3_hook = S3Hook(aws_conn_id=aws_conn_id)
        postgres_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
        
        # Read CSV header to infer columns
        csv_content = s3_hook.read_key(key=s3_key, bucket_name=s3_bucket)
        lines = csv_content.strip().split('\n')
        header = lines[0].split(',')
        
        # Create table with all columns as TEXT (safe default)
        columns_def = ',\n    '.join([f'"{col.strip()}" TEXT' for col in header])
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {columns_def}
        );
        """
        
        print(f"Creating table with SQL:\n{create_table_sql}")
        postgres_hook.run(create_table_sql)
        print(f"Table {table_name} created successfully")
        
        return table_name

    @task
    def load_s3_to_postgres_copy(s3_bucket: str, s3_key: str, table_name: str,
                                 aws_conn_id: str = 'aws_default',
                                 postgres_conn_id: str = 'postgres_default'):
        """Load CSV from S3 to PostgreSQL using COPY command"""
        
        s3_hook = S3Hook(aws_conn_id=aws_conn_id)
        postgres_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.csv') as tmp_file:
            temp_file_path = tmp_file.name
            
            try:
                # Download S3 file to temporary location
                print(f"Downloading s3://{s3_bucket}/{s3_key} to {temp_file_path}")
                s3_object = s3_hook.get_key(key=s3_key, bucket_name=s3_bucket)
                csv_content = s3_object.get()['Body'].read()
                tmp_file.write(csv_content)
                tmp_file.flush()
                
                print(f"File downloaded, size: {os.path.getsize(temp_file_path)} bytes")
                
                # Use COPY command to load data
                conn = postgres_hook.get_conn()
                cursor = conn.cursor()
                
                with open(temp_file_path, 'r') as f:
                    copy_sql = f"""
                        COPY {table_name} 
                        FROM STDIN 
                        WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')
                    """
                    cursor.copy_expert(copy_sql, f)
                
                conn.commit()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
                
                print(f"Successfully loaded {row_count} rows to table: {table_name}")
                
                return {"status": "success", "table": table_name, "rows": row_count}
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Temporary file removed")

    # Define task flow
    table_created = create_table_from_csv(
        s3_bucket='ccp-bucket-4',
        s3_key='raw.csv',
        table_name='churnprediction',
        aws_conn_id='S3_Bucket',
        postgres_conn_id='postgres_default'
    )
    
    load_data = load_s3_to_postgres_copy(
        s3_bucket='ccp-bucket-4',
        s3_key='raw.csv',
        table_name='churnprediction',
        aws_conn_id='S3_Bucket',
        postgres_conn_id='postgres_default'
    )
    
    table_created >> load_data
