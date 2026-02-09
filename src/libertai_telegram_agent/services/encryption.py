from cryptography.fernet import Fernet


def encrypt_api_key(api_key: str, encryption_key: str) -> str:
    """Encrypt an API key string. Returns encrypted string."""
    fernet = Fernet(encryption_key)
    return fernet.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str, encryption_key: str) -> str:
    """Decrypt an encrypted API key. Returns original string."""
    fernet = Fernet(encryption_key)
    return fernet.decrypt(encrypted_key.encode()).decode()
