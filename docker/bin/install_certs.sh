#!/bin/bash
set -euo pipefail

CERTS_DIR="$1"

if [ ! -d "$CERTS_DIR" ]; then
    echo "Certificate directory not found: $CERTS_DIR" >&2
    exit 1
fi

if ! compgen -G "$CERTS_DIR/*.crt" > /dev/null; then
    echo "No .crt files found in $CERTS_DIR" >&2
    exit 1
fi

if command -v keytool >/dev/null 2>&1; then
    echo "# update java ca store"
    for cert in "$CERTS_DIR"/*.crt; do
        alias=$(basename "$cert" | sed 's/.crt//')
        keytool -importcert -noprompt -trustcacerts -cacerts -alias "$alias" -file "$cert" -storepass changeit || true
    done
else
    echo "# keytool not found, skipping java ca store"
fi

echo "# update os ca store"
cp "$CERTS_DIR"/*.crt /usr/local/share/ca-certificates
update-ca-certificates
