"""
Firebase Admin SDK — Firestore connection
Place your serviceAccountKey.json in the backend/ folder.

To get serviceAccountKey.json:
  Firebase Console → Project Settings → Service Accounts → Generate New Private Key
"""
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth

_app = None

def get_firebase():
    global _app
    if _app is None:
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
