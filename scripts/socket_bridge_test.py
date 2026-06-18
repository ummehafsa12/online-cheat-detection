"""
Socket bridge sanity test for server -> student warnings (book / generic).

Usage:
  python scripts/socket_bridge_test.py --server http://127.0.0.1:5000 --student-id 21

What it does:
  - Connects a simulated student to /student with the given student_id.
  - Connects an admin to /admin.
  - Emits admin_book_detected to trigger server-side book warning flow.
  - Waits for student + admin to receive `server_object_detected` and `student_violation`.
  - Prints a short pass/fail summary.
"""

import argparse
import json
import threading
import time

import socketio


def make_client(name, namespace):
    cli = socketio.Client(logger=False, engineio_logger=False, ssl_verify=False)
    cli.name = name
    cli.received = []

    @cli.event(namespace=namespace)
    def connect():
        print(f"[{name}] connected to {namespace}")

    @cli.event(namespace=namespace)
    def disconnect():
        print(f"[{name}] disconnected from {namespace}")

    @cli.on("server_object_detected", namespace=namespace)
    def on_server_object(payload):
        cli.received.append(("server_object_detected", payload))
        print(f"[{name}] server_object_detected: {json.dumps(payload, ensure_ascii=False)}")

    @cli.on("student_violation", namespace=namespace)
    def on_student_violation(payload):
        cli.received.append(("student_violation", payload))
        print(f"[{name}] student_violation: {json.dumps(payload, ensure_ascii=False)}")

    return cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:5000", help="Server base URL")
    parser.add_argument("--student-id", default="21", help="Student ID to test with")
    parser.add_argument("--student-name", default="Test Student", help="Student name")
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for events")
    args = parser.parse_args()

    student = make_client("STUDENT", "/student")
    admin = make_client("ADMIN", "/admin")

    # Connect student with query param so server joins student room
    student_url = f"{args.server}?student_id={args.student_id}"
    transports = ["polling"]  # polling-only to avoid websocket handshake issues
    try:
        student.connect(student_url, namespaces=["/student"], socketio_path="socket.io", transports=transports, wait_timeout=8)
    except Exception as e:
        print(f"[STUDENT] polling connect failed: {e}")
        raise
    try:
        admin.connect(args.server, namespaces=["/admin"], socketio_path="socket.io", transports=transports, wait_timeout=8)
    except Exception as e:
        print(f"[ADMIN] polling connect failed: {e}")
        student.disconnect()
        raise

    # Give sockets a moment
    time.sleep(0.5)

    print("\n[TEST] Emitting admin_book_detected …")
    admin.emit(
        "admin_book_detected",
        {"student_id": str(args.student_id), "student_name": args.student_name, "label": "book"},
        namespace="/admin",
    )

    # Wait for events
    time.sleep(args.timeout)

    # Summary
    def got(ev_list, name):
        return any(ev == name for ev, _ in ev_list)

    student_got_obj = got(student.received, "server_object_detected")
    student_got_violation = got(student.received, "student_violation")
    admin_got_obj = got(admin.received, "server_object_detected")
    admin_got_violation = got(admin.received, "student_violation")

    print("\n[SUMMARY]")
    print(f" student server_object_detected: {'OK' if student_got_obj else 'MISS'}")
    print(f" student student_violation    : {'OK' if student_got_violation else 'MISS'}")
    print(f" admin   server_object_detected: {'OK' if admin_got_obj else 'MISS'}")
    print(f" admin   student_violation      : {'OK' if admin_got_violation else 'MISS'}")

    student.disconnect()
    admin.disconnect()


if __name__ == "__main__":
    main()
