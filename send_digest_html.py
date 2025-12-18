#!/usr/bin/env python3
"""
Send a lit_feed HTML digest via email.

Usage:
    python send_digest_html.py /path/to/digest_YYYY-MM-DD.html
"""

import os
import sys
import smtplib
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime


def load_html(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Digest file not found: {p}")
    return p.read_text(encoding="utf-8")


def send_email_html(
    subject: str,
    html_body: str,
    from_addr: str,
    to_addrs: list[str],
    smtp_host: str,
    smtp_port: int,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    use_starttls: bool = True,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)

    # when client does not support HTML
    msg.set_content("Your email client does not support HTML. Please view the digest in a browser.")

    # HTML content
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        if use_starttls:
            server.starttls()
        if smtp_user and smtp_password:
            server.login(smtp_user, smtp_password)
        server.send_message(msg)


def main():
    if len(sys.argv) != 2:
        print("Usage: python send_digest_html.py /path/to/digest_YYYY-MM-DD.html")
        sys.exit(1)

    digest_path = sys.argv[1]
    html = load_html(digest_path)

    # read SMTP and email config from env vars
    smtp_host = os.environ.get("LIT_SMTP_HOST")
    smtp_port = int(os.environ.get("LIT_SMTP_PORT", "587"))
    smtp_user = os.environ.get("LIT_SMTP_USER")
    smtp_password = os.environ.get("LIT_SMTP_PASS")
    from_addr = os.environ.get("LIT_FROM")
    to_raw = os.environ.get("LIT_TO")  # multiple recipients separated by commas
    use_starttls = os.environ.get("LIT_SMTP_STARTTLS", "1") == "1"

    if not (smtp_host and from_addr and to_raw):
        print("Error: please set LIT_SMTP_HOST, LIT_FROM, LIT_TO env vars.")
        sys.exit(1)

    to_addrs = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

    today = datetime.now().strftime("%Y-%m-%d")
    subject = os.environ.get("LIT_SUBJECT", f"[LitFeed] Recent papers digest ({today})")

    send_email_html(
        subject=subject,
        html_body=html,
        from_addr=from_addr,
        to_addrs=to_addrs,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        use_starttls=use_starttls,
    )

    print(f"Digest sent to: {', '.join(to_addrs)}")


if __name__ == "__main__":
    main()
