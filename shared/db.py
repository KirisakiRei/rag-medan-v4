"""
RAG Medan v3 - Shared Database Utilities
"""
import mysql.connector
from mysql.connector import Error
from config import config


def _get_connection():
    """Get MySQL database connection."""
    return mysql.connector.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        database=config.DB_DATABASE,
        user=config.DB_USERNAME,
        password=config.DB_PASSWORD,
        autocommit=True
    )


def get_variable(name: str) -> str | None:
    """Get content from `variables` table by name. Returns None if not found."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM variables WHERE name = %s AND deleted_at IS NULL LIMIT 1",
            (name,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return row[0]
        return None
    except Error:
        return None


def execute_query(query: str, params: tuple = None) -> list:
    """Execute SELECT query and return results."""
    try:
        conn = _get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params or ())
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Error:
        return []


def execute_update(query: str, params: tuple = None) -> bool:
    """Execute INSERT/UPDATE/DELETE query."""
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(query, params or ())
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Error:
        return False
