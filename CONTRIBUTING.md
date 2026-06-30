# Contributing to harinezumigel-llm-stack

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.10+
- Docker with NVIDIA Container Toolkit
- pylint (for code quality)

### Local Setup

```bash
# Clone your fork
git clone https://github.com/Harinezumigel/harinezumigel-llm-stack.git
cd harinezumigel-llm-stack

# Install dependencies
pip install pyyaml pylint

# Set up test environment
cp .env.example /opt/litellm/.env
cp config.yaml.example /opt/litellm/config.yaml
# Edit files with your local paths
```

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 88 characters (Black formatter compatible)
- Use docstrings for all public functions and classes

### Docstring Format

```python
def function_name(arg1: str, arg2: int) -> bool:
    """Brief description of function.

    More detailed description if needed. Explain the purpose,
    not just what the code does.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Safety:
        Include safety notes for destructive operations
    """
```

### Code Organization

- Keep functions focused and single-purpose
- Group related functions in sections with clear headers
- Use dataclasses for configuration structures
- Avoid global state (except `APP` config)

## Testing

### Manual Testing Checklist

Before submitting, test:

- [ ] `--list` displays models correctly
- [ ] Starting a model (new container)
- [ ] Reusing an existing container
- [ ] `--recreate` flag works
- [ ] `--dry-run` shows correct commands without executing
- [ ] Log viewing (`--logs`)
- [ ] Stopping containers (`--stop`)
- [ ] LiteLLM start/stop
- [ ] Port conflict detection
- [ ] Invalid model name handling
- [ ] Missing configuration file handling

### Test with Different Scenarios

```bash
# Test basic functionality
harinezumigel-llm-stack --list
harinezumigel-llm-stack test-model --dry-run

# Test error handling
harinezumigel-llm-stack nonexistent-model
harinezumigel-llm-stack model-name --port 99999

# Test safety features
harinezumigel-llm-stack model-name --recreate --dry-run
harinezumigel-llm-stack --stop all --dry-run
```

## Safety Considerations

This script manages Docker containers and system processes. When contributing:

### Critical Safety Rules

1. **Always respect `dry_run` flag**: Any destructive operation must check `dry_run`
2. **Container scoping**: Only operate on containers matching `vllm-{model_name}-*` pattern
3. **Explicit flags**: Destructive operations require explicit user flags (`--recreate`, `--stop`)
4. **No file modifications**: Script should only read files, never write/delete
5. **Process safety**: When killing processes, use specific pattern matching

### Before Adding Destructive Operations

Ask yourself:
- Does this respect `--dry-run`?
- Is there clear user intent (explicit flag)?
- Are there safety checks to prevent accidents?
- Is the operation scoped appropriately?
- Is there informative output before execution?

## Feature Guidelines

### Adding New Features

1. **Maintain backward compatibility**: Don't break existing workflows
2. **Follow existing patterns**: Use similar code structure to current implementation
3. **Add documentation**: Update README.md and docstrings
4. **Consider dry-run**: New operations should support `--dry-run`
5. **Error handling**: Provide clear error messages

### Good Feature Examples

✅ Add support for custom Docker networks
✅ Add JSON output mode for scripting
✅ Add health check command
✅ Support additional vLLM parameters

### Features to Avoid

❌ Automatic model downloading (security risk)
❌ Modifying config files programmatically (user should control)
❌ Silent destructive operations (always require explicit flags)
❌ Hardcoding configuration values

## Pull Request Process

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All functions have docstrings
- [ ] Destructive operations respect `--dry-run`
- [ ] Changes are tested manually
- [ ] README.md updated if needed
- [ ] No hardcoded values (use config files)
- [ ] Error messages are clear and helpful

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup

## Testing
Describe how you tested the changes

## Related Issues
Fixes #(issue number)

## Additional Notes
Any other relevant information
```

## Common Contribution Areas

### Easy Contributions (Good First Issues)

- Improve error messages
- Add more detailed docstrings
- Fix typos in documentation
- Add usage examples to README
- Improve help text formatting

### Medium Contributions

- Add new CLI flags for existing vLLM parameters
- Improve model directory fuzzy matching
- Add validation for config.yaml
- Add container health checks
- Improve log formatting

### Advanced Contributions

- Implement subcommand-based CLI (see ARGS_SUGGESTIONS.md)
- Add Docker Compose support
- Add Kubernetes deployment templates
- Implement model status dashboard
- Add metrics collection

## Bug Reports

### Good Bug Reports Include

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Exact commands to trigger the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: OS, Python version, Docker version
6. **Logs**: Relevant error messages or logs
7. **Configuration**: Sanitized .env and config.yaml (remove secrets!)

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**Steps to Reproduce**
1. Run command: `python setupllm.py ...`
2. Observe behavior

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10.12
- Docker: 24.0.5
- Script version: commit hash or release

**Logs**
```
Paste relevant logs here
```

**Configuration**
(Sanitized - remove API keys!)
```yaml
Relevant config.yaml snippet
```
```

## Questions and Support

- 💬 Use [Discussions](https://github.com/Harinezumigel/harinezumigel-llm-stack/discussions) for questions
- 🐛 Use [Issues](https://github.com/Harinezumigel/harinezumigel-llm-stack/issues) for bugs
- 📖 Check [README.md](README.md) and [SAFETY_ANALYSIS.md](SAFETY_ANALYSIS.md) first

## Code Review Process

### What We Look For

1. **Correctness**: Does it work as intended?
2. **Safety**: Could it cause data loss or unexpected behavior?
3. **Code quality**: Is it readable and maintainable?
4. **Documentation**: Are changes documented?
5. **Testing**: Has it been tested?

### Review Timeline

- Initial review: Usually within 1 week
- Follow-up reviews: Within 3-5 days
- Merge: After approval and CI passes (if applicable)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- README.md (significant contributions)
- Release notes
- Git commit history

Thank you for contributing! 🎉
