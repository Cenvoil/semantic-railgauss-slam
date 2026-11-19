# Release notes

This is the public paper snapshot of RailGauss-OSDaR23. It is derived from the
private development repository and sanitized for release:

- server usernames and absolute `/home/...` paths replaced with `<PROJECT_ROOT>` / `<user>`;
- local Windows paths replaced with `<PROJECT_ROOT>`;
- server IPs and hostnames removed;
- `__pycache__`, checkpoints, datasets, logs, and private checklists excluded.

The code is a research archive, not a turnkey package. Set the data paths in the
configs or environment variables before running. See `paper/README.md` for the
paper-specific add-ons and `README.md` for the method/protocol documentation.
