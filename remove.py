import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

# 删除特定 thread_id 的记录
cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", ("user_123",))
conn.commit()
conn.close()