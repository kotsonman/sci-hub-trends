import sqlite3

def get_data_from_database(db_path):
    """Извлечение данных из базы данных SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM articles") # Замените 'articles' на название вашей таблицы
    titles = [row[0] for row in cursor.fetchall()]
    conn.close()
    return titles
