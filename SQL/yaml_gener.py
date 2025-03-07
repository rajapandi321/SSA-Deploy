import os
import yaml
from sqlalchemy import create_engine, inspect
from sqlalchemy.schema import MetaData
from dotenv import load_dotenv
import urllib


def generate_yaml(conn):
    """
    Generate a YAML representation of the database schema.
    
    Args:
        conn (str): Database connection engine
    
    Returns:
        str: YAML representation of database schema
    """
    if conn:
        print("Generating Yaml File")

    metadata = MetaData()

    try:
        metadata.reflect(bind=conn)
        inspector = inspect(conn)
    except Exception as e:
        print(f"Error reflecting database schema: {e}")
        return None

    table_data = {}

    for table_name, table in metadata.tables.items():
        table_info = {
            'columns': [],
            'primary_key': [],
            'foreign_keys': []
        }

        for column in table.columns:
            column_info = {
                'name': str(column.name),
                'type': str(column.type),
                'nullable': column.nullable,
                'default': column.default.arg if column.default else None
            }
            table_info['columns'].append(column_info)

        # Get primary key information
        primary_key = inspector.get_pk_constraint(table_name)
        if primary_key['constrained_columns']:
            table_info['primary_key'] = primary_key['constrained_columns']

        # Get foreign key information
        foreign_keys = inspector.get_foreign_keys(table_name)
        for fk in foreign_keys:
            fk_info = {
                'constrained_columns': fk['constrained_columns'],
                'referred_table': fk['referred_table'],
                'referred_columns': fk['referred_columns']
            }
            table_info['foreign_keys'].append(fk_info)

        # Add table information to the main dictionary
        table_data[table_name] = table_info

    # Serialize to YAML
    yaml_output = yaml.dump(table_data, default_flow_style=False)
    
    return yaml_output

def save_yaml_to_file(yaml_content, filename='database_schema.yaml'):
    """
    Save the YAML content to a file.
    
    Args:
        yaml_content (str): YAML representation of database schema
        filename (str, optional): Output filename. Defaults to 'database_schema.yaml'.
    """
    with open(filename, 'w') as file:
        file.write(yaml_content)
        
# if __name__ == "__main__":
#     user = 'azureuser'
#     password = 'Optisol2020'
#     server_name = 'azure-tese-db.database.windows.net'
#     database = 'self-service-poc'

#     params = urllib.parse.quote_plus(
#         f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#         f"SERVER={server_name};"
#         f"DATABASE={database};"
#         f"UID={user};"
#         f"PWD={password};"
#         "Encrypt=yes;"
#         "TrustServerCertificate=no;"
#         "Connection Timeout=30"
# )

#     db_uri=f"mssql+pyodbc:///?odbc_connect={params}"
#     engine = create_engine(db_uri)
#     yaml_output=generate_yaml(engine)
#     save_yaml_to_file(yaml_output)
    
    