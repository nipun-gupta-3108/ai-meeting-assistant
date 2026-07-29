"""Authentication business logic: signup, login, password hashing.

core/user_repository.py owns the users table (SQL only, no validation, no
password logic). This module owns everything above that layer — password
hashing/verification, email normalization, input validation, and the
typed exceptions the UI (streamlit_app.py) catches to show user-facing
messages. The repository layer never needs to know any of these rules.
"""

import re

import bcrypt

from core.user_repository import (
    create_user,
    get_user_by_email,
    initialize_users_table,
)

MIN_PASSWORD_LENGTH = 8

# Deliberately simple — this only needs to catch obviously malformed
# input (missing "@", missing domain), not fully validate per RFC 5322.
# The real check that matters is whether the email is unique, which the
# database's UNIQUE constraint enforces regardless of this regex.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Base class for user-facing authentication errors.

    streamlit_app.py can catch this alone if it just wants a generic
    error message, or catch the specific subclasses below when it needs
    to react differently (e.g. pre-filling the email field on a
    duplicate-account error).
    """


class EmailAlreadyExistsError(AuthError):
    """Raised on signup when the (normalized) email is already registered."""


class InvalidCredentialsError(AuthError):
    """Raised on login when the email/password combination doesn't match.

    Deliberately does not distinguish "no such user" from "wrong
    password" — doing so would let a caller enumerate which emails are
    registered by observing which error comes back.
    """


class ValidationError(AuthError):
    """Raised when signup input fails basic validation (empty name,
    malformed email, short password, mismatched confirmation)."""


def normalize_email(email: str) -> str:
    """Lowercase + strip an email address for consistent storage/lookup.

    Applied identically on both signup and login so "User@Example.com"
    and "user@example.com" are always treated as the same account.
    """
    return email.strip().lower()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # A malformed stored hash should never happen in practice, but
        # fail closed (reject the login) rather than raising up into the
        # UI layer.
        return False


def _validate_signup_input(
    name: str, email: str, password: str, confirm_password: str
) -> None:
    if not name.strip():
        raise ValidationError("Please enter your name.")

    if not _EMAIL_RE.match(email.strip()):
        raise ValidationError("Please enter a valid email address.")

    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )

    if password != confirm_password:
        raise ValidationError("Passwords do not match.")


def _public_user(user: dict) -> dict:
    """Strip password_hash before a user dict leaves this module.

    Nothing outside core/auth.py and core/user_repository.py should ever
    see a password hash — this is the one seam that enforces that.
    """
    return {key: value for key, value in user.items() if key != "password_hash"}


def signup(name: str, email: str, password: str, confirm_password: str) -> dict:
    """Validate input, create a new user, and return the public user dict.

    Raises ValidationError for bad input and EmailAlreadyExistsError if
    the normalized email is already registered. The returned dict never
    contains "password_hash" (see _public_user).
    """
    _validate_signup_input(name, email, password, confirm_password)

    normalized_email = normalize_email(email)

    if get_user_by_email(normalized_email) is not None:
        raise EmailAlreadyExistsError("An account with this email already exists.")

    password_hash = _hash_password(password)
    user = create_user(name.strip(), normalized_email, password_hash)

    return _public_user(user)


def login(email: str, password: str) -> dict:
    """Verify credentials and return the public user dict on success.

    Raises InvalidCredentialsError on any mismatch (unknown email or
    wrong password) — see that exception's docstring for why the two
    cases aren't distinguished from the caller's point of view.
    """
    normalized_email = normalize_email(email)
    user = get_user_by_email(normalized_email)

    if user is None or not _verify_password(password, user["password_hash"]):
        raise InvalidCredentialsError("Incorrect email or password.")

    return _public_user(user)


def ensure_users_table() -> None:
    """Thin re-export so callers only need `core.auth` for startup setup.

    Idempotent, matching core.meeting_repository.initialize_database()'s
    pattern — safe to call on every process start.
    """
    initialize_users_table()
