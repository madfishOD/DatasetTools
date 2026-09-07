"""Credential lifecycle tests use a private fake store, never personal credentials."""
import unittest
from unittest.mock import patch
import credentials


class Store:
    def __init__(self): self.values = {}
    def get_password(self, service, account): return self.values.get((service, account))
    def set_password(self, service, account, token): self.values[service, account] = token
    def delete_password(self, service, account): del self.values[service, account]


class CredentialTests(unittest.TestCase):
    def test_save_reload_replace_and_remove(self):
        backend = Store()
        with patch.object(credentials, 'native_store', return_value=backend):
            self.assertEqual(credentials.load_token(), ('', ''))
            credentials.save_token('hf_fakeOne')
            self.assertEqual(credentials.load_token(), ('hf_fakeOne', ''))
            credentials.save_token('hf_fakeTwo')
            self.assertEqual(credentials.load_token(), ('hf_fakeTwo', ''))
            credentials.save_token('')
            self.assertEqual(credentials.load_token(), ('', ''))
            credentials.save_token('')
        self.assertFalse(backend.values)

    def test_locked_store_reports_error_without_plaintext_fallback(self):
        with patch.object(credentials, 'native_store', side_effect=RuntimeError('private backend detail')):
            token, notice = credentials.load_token()
            self.assertEqual(token, '')
            self.assertIn('locked', notice)
            with self.assertRaises(RuntimeError) as caught:
                credentials.save_token('hf_fakeSecret')
            self.assertNotIn('hf_fakeSecret', str(caught.exception))
            self.assertNotIn('private backend detail', str(caught.exception))

    def test_platform_selection_does_not_use_arbitrary_keyring_backend(self):
        with patch.object(credentials.sys, 'platform', 'linux'):
            with self.assertRaises(RuntimeError): credentials.native_store()


if __name__ == '__main__': unittest.main()
