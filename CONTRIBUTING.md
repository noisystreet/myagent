# Contributing

Thank you for considering contributing to myagent!

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes (follow Conventional Commits format)
4. Make sure all tests pass: `make test`
5. Open a Pull Request

## Commit Format

```
<type>(<scope>): <short description>

<detailed description (optional)>
```

See [AGENTS.md](AGENTS.md#提交规范conventional-commits) for available types and scopes.

## Development Setup

```bash
make install    # Install dependencies
make test       # Run tests
make lint       # Lint code
make complexity # Check complexity
```

## Security Issues

Please **do not** report security vulnerabilities in public issues.
See [SECURITY.md](SECURITY.md) for our disclosure policy.

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/).
