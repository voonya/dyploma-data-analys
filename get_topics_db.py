import psycopg2
import pandas as pd
# Establish connection to PostgreSQL
conn = psycopg2.connect(
    dbname="dyploma-db",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

def get_current_topics_db():
    sql_query_topic_creation = "SELECT * FROM public.\"TopicCreation\" ORDER BY \"createdAt\" DESC LIMIT 1;"
    df_topic_creation = pd.read_sql_query(sql_query_topic_creation, conn)

    sql_query_topics = f"SELECT * FROM public.\"Topic\" WHERE \"topicCreationId\" = \'{df_topic_creation['id'][0]}\' ORDER BY \"topicDataId\" ASC;"
    df_topics = pd.read_sql_query(sql_query_topics, conn)

    return df_topics