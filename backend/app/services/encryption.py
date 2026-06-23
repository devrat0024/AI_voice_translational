import os
import base64
import hashlib
from cryptography.fernet import Fernet
from backend.app.config import SECRET_KEY

# Check if custom ENCRYPTION_KEY is provided, otherwise derive a stable key from SECRET_KEY
_raw_key = os.getenv("ENCRYPTION_KEY", "")
if _raw_key:
    try:
        # Check if it's already a valid Fernet key
        _fernet = Fernet(_raw_key.encode())
        _key_bytes = _raw_key.encode()
    except Exception:
        # If not base64 valid, derive a valid base64 key from it
        _key_bytes = base64.urlsafe_b64encode(hashlib.sha256(_raw_key.encode()).digest())
else:
    # Stable fallback derived from the application SECRET_KEY
    _key_bytes = base64.urlsafe_b64encode(hashlib.sha256(SECRET_KEY.encode()).digest())

cipher_suite = Fernet(_key_bytes)


def encrypt_data(plain_text: str) -> str:
    """Encrypts plaintext string to base64 AES-256 ciphertext."""
    if not plain_text:
        return ""
    encrypted_bytes = cipher_suite.encrypt(plain_text.encode("utf-8"))
    return encrypted_bytes.decode("utf-8")


def decrypt_data(cipher_text: str) -> str:
    """Decrypts base64 AES-256 ciphertext string to plaintext."""
    if not cipher_text:
        return ""
    try:
        decrypted_bytes = cipher_suite.decrypt(cipher_text.encode("utf-8"))
        return decrypted_bytes.decode("utf-8")
    except Exception:
        # Return fallback value or empty if decryption fails
        return "[Decryption Error: Invalid Key or Corrupted Data]"
