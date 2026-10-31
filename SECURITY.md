# Security

## Reporting

Email the maintainer listed on the GitHub profile for `wwylon/upscale`, or open a **private** GitHub security advisory if enabled. Do not post exploit details in public issues.

## Deployment notes

Default web bind is loopback (`127.0.0.1`). Treat uploaded images and model paths as untrusted input on shared hosts. Model weights are downloaded at runtime — verify checksums where the CLI provides them before redistributing checkpoints.
