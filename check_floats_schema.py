from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

from ingest.db_handler import SupabaseHandler
from sqlalchemy import text

db = SupabaseHandler()
conn = db.engine.connect()
result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'floats'"))
columns = [row[0] for row in result.fetchall()]
print('Floats table columns:', columns)

# Also get a sample row to see the structure
result = conn.execute(text("SELECT * FROM floats LIMIT 1"))
sample_row = result.fetchone()
if sample_row:
    print('Sample row keys:', list(sample_row._mapping.keys()))

conn.close()
db.close()