FROM astrocrpublic.azurecr.io/runtime:3.1-3

RUN pip3 install 'apache-airflow[amazon]'
RUN pip3 install 'apache-airflow[postgres]'
RUN pip3 install 'apache-airflow-providers-postgres'