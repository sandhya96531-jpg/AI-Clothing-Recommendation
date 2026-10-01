import sqlite3
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Database directory
DATABASE_DIR = BASE_DIR / "storage" / "database"

# Create directory if it does not exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# SQLite database path
DATABASE_PATH = DATABASE_DIR / "ai_style.db"


def get_connection():
    """
    Create and return a connection to the SQLite database.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    # Allows database rows to behave like dictionaries
    connection.row_factory = sqlite3.Row

    # Enable foreign-key relationships
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """
    Create all required database tables.
    """

    connection = get_connection()

    cursor = connection.cursor()


    # =========================
    # USERS TABLE
    # =========================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            full_name TEXT NOT NULL,

            email TEXT NOT NULL UNIQUE,

            password_hash TEXT NOT NULL,

            category TEXT,

            age INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )
    """)


    # =========================
    # USER STYLE PROFILES
    # =========================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS style_profiles (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL UNIQUE,

            skin_tone TEXT,

            undertone TEXT,

            body_shape TEXT,

            body_proportions TEXT,

            height_cm REAL,

            preferred_style TEXT,

            preferred_occasion TEXT,

            preferred_colours TEXT,

            budget REAL,

            analysis_status TEXT DEFAULT 'pending',

            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE

        )
    """)


    # =========================
    # CLOTHING TABLE
    # =========================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clothing (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            outfit_id TEXT NOT NULL UNIQUE,

            outfit_name TEXT NOT NULL,

            category TEXT NOT NULL,

            clothing_type TEXT,

            colour TEXT,

            pattern TEXT,

            style TEXT,

            occasion TEXT,

            fit TEXT,

            suitable_body_shapes TEXT,

            suitable_skin_tones TEXT,

            suitable_undertones TEXT,

            suitable_proportion_types TEXT,

            min_height_cm REAL,

            max_height_cm REAL,

            image_path TEXT,

            tryon_image_path TEXT,

            product_store TEXT,

            price REAL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )
    """)


    # =========================
    # FAVOURITES TABLE
    # =========================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS favourites (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            clothing_id INTEGER NOT NULL,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(user_id, clothing_id),

            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE,

            FOREIGN KEY (clothing_id)
                REFERENCES clothing(id)
                ON DELETE CASCADE

        )
    """)


    # =========================
    # TRY-ON RESULTS TABLE
    # =========================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tryon_results (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            clothing_id INTEGER NOT NULL,

            original_photo_path TEXT,

            result_photo_path TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE,

            FOREIGN KEY (clothing_id)
                REFERENCES clothing(id)
                ON DELETE CASCADE

        )
    """)


    connection.commit()

    connection.close()


if __name__ == "__main__":

    initialize_database()

    print("Database initialized successfully.")
    print(f"Database location: {DATABASE_PATH}")