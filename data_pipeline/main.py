import argparse
import sys
import logging
from data_pipeline.pipeline.config import init_directories, DB_PATH
from data_pipeline.pipeline.storage import init_db, get_connection, get_stats
from data_pipeline.pipeline.etl import run_etl_pipeline

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="AI Scribe Data Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: init-db
    subparsers.add_parser("init-db", help="Initialize database schemas and directories")

    # Command: run-etl
    subparsers.add_parser("run-etl", help="Run the ETL ingestion & preprocessing pipeline")

    # Command: show-stats
    subparsers.add_parser("show-stats", help="Show database contents and records count statistics")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Execute a raw SQL query on the database")
    query_parser.add_argument("sql", type=str, help="SQL query string (e.g. 'SELECT * FROM patients')")

    args = parser.parse_args()

    if args.command == "init-db":
        logger.info("Initializing directories...")
        init_directories()
        logger.info("Initializing database...")
        init_db()
        logger.info("Setup complete.")
        
    elif args.command == "run-etl":
        logger.info("Starting ETL Pipeline run...")
        stats = run_etl_pipeline()
        print("\n=== ETL RUN COMPLETED ===")
        print(f"Total Patients: {stats.get('patients_count', 0)}")
        print(f"Audio Records status counts: {stats.get('audio_counts', {})}")
        print(f"Medical Documents status counts: {stats.get('document_counts', {})}")
        print("=========================")

    elif args.command == "show-stats":
        if not DB_PATH.exists():
            logger.error("Database has not been initialized. Please run: python main.py init-db")
            sys.exit(1)
        conn = get_connection()
        try:
            stats = get_stats(conn)
            print("\n=== DATABASE STATISTICS ===")
            print(f"Database File: {DB_PATH.resolve()}")
            print(f"Total Patients: {stats.get('patients_count', 0)}")
            print("\nAudio Records status counts:")
            for status, count in stats.get('audio_counts', {}).items():
                print(f"  - {status}: {count}")
            print("\nMedical Documents status counts:")
            for status, count in stats.get('document_counts', {}).items():
                print(f"  - {status}: {count}")
            print("===========================")
        finally:
            conn.close()

    elif args.command == "query":
        if not DB_PATH.exists():
            logger.error("Database has not been initialized. Please run: python main.py init-db")
            sys.exit(1)
        import pandas as pd
        conn = get_connection()
        try:
            df = pd.read_sql_query(args.sql, conn)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print("\n=== QUERY RESULTS ===")
            print(df)
            print("=====================")
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
        finally:
            conn.close()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
