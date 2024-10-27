import sqlite3
import csv

def create_and_insert_db(csv_file):
    # Conectar a la base de datos SQLite (o crearla si no existe)
    conn = sqlite3.connect('customer_support.db')
    cursor = conn.cursor()
    
    # Crear la tabla
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client TEXT,
            ticket_number INTEGER,
            issue_type TEXT,
            description TEXT
        )
    ''')
    
    # Leer datos del archivo CSV e insertarlos en la base de datos
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            cursor.execute('''
                INSERT INTO tickets (client, ticket_number, issue_type, description)
                VALUES (?, ?, ?, ?)
            ''', (row['Client'], row['Ticket Number'], row['Issue Type'], row['Description']))
    
    # Confirmar los cambios y cerrar la conexión
    conn.commit()
    conn.close()