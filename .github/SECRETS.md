# GitHub Secrets Configuration

This document describes the required secrets for GitHub Actions workflows.

## Required Secrets

### For Docker Hub Deployment (Optional)

If you want to enable Docker Hub deployment, add these secrets to your GitHub repository:

1. Go to your repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Add the following repository secrets:

#### `DOCKER_USERNAME`
- **Description**: Your Docker Hub username
- **Example**: `your-dockerhub-username`
- **Required for**: Docker image push to Docker Hub

#### `DOCKER_PASSWORD`
- **Description**: Your Docker Hub password or access token
- **Example**: `your-dockerhub-password-or-token`
- **Required for**: Docker image push to Docker Hub

### For PyPI Release (Optional)

If you want to enable PyPI releases, add these secrets:

#### `PYPI_API_TOKEN`
- **Description**: PyPI API token for uploading packages
- **Example**: `pypi-AgEIcHlwaS5vcmcC...`
- **Required for**: Automatic PyPI package uploads

## Workflow Behavior

### Without Secrets
- **CI/CD**: All tests and builds will run normally
- **Docker**: Docker images will be built but not pushed (test only)
- **PyPI**: No automatic releases will be triggered

### With Secrets
- **Docker**: Images will be built and pushed to Docker Hub
- **PyPI**: Automatic releases will be triggered on version tags

## Setting Up Secrets

### Docker Hub
1. Create a Docker Hub account at https://hub.docker.com
2. Generate an access token in your Docker Hub settings
3. Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets

### PyPI
1. Create a PyPI account at https://pypi.org
2. Generate an API token in your PyPI account settings
3. Add `PYPI_API_TOKEN` secret

## Security Notes

- Never commit secrets to the repository
- Use access tokens instead of passwords when possible
- Regularly rotate your tokens
- Use repository-level secrets, not organization-level secrets for public repos

## Testing Without Secrets

The workflows are designed to work without secrets:

```bash
# Test locally
./scripts/run_tests.sh

# Build locally
python -m build

# Test Docker build locally
docker build -t alphascrabble:test .
```

## Troubleshooting

### Docker Login Failed
- Check that `DOCKER_USERNAME` and `DOCKER_PASSWORD` are set correctly
- Verify your Docker Hub credentials
- Ensure the repository has push permissions

### PyPI Upload Failed
- Check that `PYPI_API_TOKEN` is valid
- Verify the token has upload permissions
- Ensure the package version is unique

### Workflow Skipped
- Check that secrets are set in the correct repository
- Verify the workflow conditions are met
- Check the workflow logs for specific error messages
