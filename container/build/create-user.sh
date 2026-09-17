# Free up UID 1000
userdel -r ubuntu 2>/dev/null || true

# Create user matching host group and user
groupadd -f -g "$XGID" user
useradd -m -u "$XUID" -g "$XGID" -s /bin/bash user
# chown user:user /app
