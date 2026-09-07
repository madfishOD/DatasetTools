"""Application-specific credentials in native OS storage; no plaintext fallback."""
import sys

SERVICE = 'DatasetTools.HuggingFace'
ACCOUNT = 'access-token'


def native_store():
    if sys.platform == 'darwin':
        from keyring.backends.macOS import Keyring
        return Keyring()
    if sys.platform == 'win32':
        from keyring.backends.Windows import WinVaultKeyring
        return WinVaultKeyring()
    raise RuntimeError('Persistent credentials are supported on macOS and Windows only.')


def load_token():
    try:
        return native_store().get_password(SERVICE, ACCOUNT) or '', ''
    except Exception:
        return '', 'System credential storage is unavailable or locked. Open Hugging Face token settings to retry.'


def save_token(token):
    try:
        backend = native_store()
        if token:
            backend.set_password(SERVICE, ACCOUNT, token)
        elif backend.get_password(SERVICE, ACCOUNT) is not None:
            backend.delete_password(SERVICE, ACCOUNT)
    except Exception:
        raise RuntimeError('Unable to update system credential storage. Unlock it or allow access, then retry. No token was saved to a configuration file.') from None
