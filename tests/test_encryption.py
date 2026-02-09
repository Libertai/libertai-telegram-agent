from cryptography.fernet import Fernet

from libertai_telegram_agent.services.encryption import decrypt_api_key, encrypt_api_key


ENCRYPTION_KEY = Fernet.generate_key().decode()


class TestEncryptDecrypt:
    """Encrypt then decrypt returns original value."""

    def test_round_trip(self):
        original = "sk-test-api-key-abc123"
        encrypted = encrypt_api_key(original, ENCRYPTION_KEY)
        decrypted = decrypt_api_key(encrypted, ENCRYPTION_KEY)

        assert decrypted == original

    def test_two_encryptions_differ(self):
        """Two encryptions of the same value produce different ciphertexts (Fernet includes timestamp)."""
        api_key = "sk-test-api-key-abc123"
        encrypted_a = encrypt_api_key(api_key, ENCRYPTION_KEY)
        encrypted_b = encrypt_api_key(api_key, ENCRYPTION_KEY)

        assert encrypted_a != encrypted_b
