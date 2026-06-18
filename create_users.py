import pymysql.cursors
from app import _hash_password_bcrypt

# Connect to the local MySQL instance
connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='',
    db='examproctordb',
    port=3306,
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True
)

try:
    with connection.cursor() as cursor:
        # Create Student Account
        student_email = 'student@test.com'
        student_pass = 'password123'
        student_hash = _hash_password_bcrypt(student_pass)
        
        # Ignore the duplicate entry exception from the previous run
        try:
            cursor.execute(
                "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                ('Test Student', student_email, student_hash, 'STUDENT')
            )
            print(f"Created Student Account: {student_email} / {student_pass}")
        except pymysql.err.IntegrityError:
            print(f"Student Account already exists: {student_email} / {student_pass}")

        # Create Admin Account
        admin_email = 'admin@test.com'
        admin_pass = 'admin123'
        admin_hash = _hash_password_bcrypt(admin_pass)
        
        try:
            cursor.execute(
                "INSERT INTO students (Name, Email, Password, Role) VALUES (%s, %s, %s, %s)",
                ('Test Admin', admin_email, admin_hash, 'ADMIN')
            )
            print(f"Created Admin Account: {admin_email} / {admin_pass}")
        except pymysql.err.IntegrityError:
            print(f"Admin Account already exists: {admin_email} / {admin_pass}")

except Exception as e:
    print(f"Error: {e}")
finally:
    connection.close()

