import json


def print_session(session) -> None:
    """友好打印 session 对话内容"""
    session_dict = session.to_dict()
    messages = session_dict.get("state", {}).get("in_memory", {}).get("messages", [])
    role_labels = {"user": "👤 User", "assistant": "🤖 Assistant", "tool": "🔧 Tool"}
    print(f"\n{'='*50}")
    print(f"  Session ID: {session_dict.get('session_id')}")
    print(f"  Messages: {len(messages)}")
    print(f"{'='*50}")
    for idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        author = msg.get("author_name", "")
        label = role_labels.get(role, role)
        if author:
            label = f"{label} ({author})"
        print(f"\n[{idx+1}] {label}")
        for c in msg.get("contents", []):
            ctype = c.get("type", "")
            if ctype == "text":
                print(f"    {c['text']}")
            elif ctype == "function_call":
                args = json.loads(c.get("arguments", "{}"))
                print(f"    Call: {c['name']}({', '.join(f'{k}={v}' for k, v in args.items())})")
            elif ctype == "function_result":
                print(f"    Result: {c.get('result', '')}")
    print(f"\n{'='*50}")
