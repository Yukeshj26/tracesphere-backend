"""
Firebase Admin SDK — Firestore connection
On Render: set FIREBASE_SERVICE_ACCOUNT_JSON environment variable with the contents of serviceAccountKey.json
Locally: place serviceAccountKey.json in the backend/ folder
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth

_app = None

def get_firebase():
    global _app
    if _app is None:
        key_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if key_json:
            # Render: read from environment variable
            cred = credentials.Certificate(json.loads(key_json))
        else:
            # Local: read from file
            key_path = os.path.join(os.path.dirname(__file__), "..", "serviceAccountKey.json")
            cred = credentials.Certificate(key_path)
        _app = firebase_admin.initialize_app(cred)
    return _app

def get_db() -> firestore.Client:
    get_firebase()
    return firestore.client()

def get_auth():
    get_firebase()
    return auth
