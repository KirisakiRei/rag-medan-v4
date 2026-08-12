"""
RAG Medan v3 - Shared Database Utilities
"""
import mysql.connector
from mysql.connector import pooling
from mysql.connector import Error
from config import config


_POOL_SIZE = 5
_pool = None


def _get_pool():
    """Lazy-init a MySQL connection pool (koneksi di-reuse antar call)."""
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="rag_pool",
            pool_size=_POOL_SIZE,
            pool_reset_session=False,
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_DATABASE,
            user=config.DB_USERNAME,
            password=config.DB_PASSWORD,
            autocommit=True,
        )
    return _pool


def _get_connection():
    """
    Get MySQL connection from pool (fallback: direct connect).

    - `pool_reset_session=False` menghindari reset session tambahan tiap
      peminjaman koneksi.
    - `ping(reconnect=True)` memastikan koneksi hasil reuse masih hidup
      (mencegah "MySQL server has gone away" setelah idle).
    - Fallback ke koneksi langsung jika pool habis/error agar tidak ada
      request yang gagal hanya karena pool exhaustion.
    """
    conn = None
    try:
        conn = _get_pool().get_connection()
        conn.ping(reconnect=True)
        return conn
    except Error:
        if conn is not None:
            try:
                conn.close()
            except Error:
                pass
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
