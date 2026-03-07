chat_sessions = {}

def get_history(session_id):
    return chat_sessions.get(session_id, [])

def save_message(session_id, role, content):
    chat_sessions.setdefault(session_id, []).append({
        "role": role,
        "content": content
    })