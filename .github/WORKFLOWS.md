# GitHub Workflows Configuration

This document explains the GitHub Actions workflows and how to configure them.

## Available Workflows

### 1. Simple CI (`.github/workflows/simple.yml`) ✅ **ACTIVE**
- **Purpose**: Basic testing without complex dependencies
- **Triggers**: Push to main/develop, pull requests
- **Features**: 
  - Python 3.11 setup
  - Basic dependency installation
  - Import testing
  - Basic functionality testing
- **Status**: ✅ Working, no secrets required

### 2. Test Suite (`.github/workflows/test.yml`) ✅ **ACTIVE**
- **Purpose**: Comprehensive testing with pytest
- **Triggers**: Push to main/develop, pull requests
- **Features**:
  - Multiple Python versions (3.10, 3.11, 3.12)
  - System dependencies installation
  - Full test suite execution
  - Import testing
- **Status**: ✅ Working, no secrets required

### 3. Build (`.github/workflows/build.yml`) ✅ **ACTIVE**
- **Purpose**: Package building and Docker image creation
- **Triggers**: Push to main, pull requests
- **Features**:
  - Python package building
  - Docker image building (no push)
  - Artifact upload
- **Status**: ✅ Working, no secrets required

### 4. Deploy (`.github/workflows/deploy.yml`) ⚠️ **CONDITIONAL**
- **Purpose**: Deployment to Docker Hub and Colab
- **Triggers**: Push to main, manual dispatch
- **Features**:
  - Colab deployment (always works)
  - Docker Hub deployment (requires secrets)
- **Status**: ⚠️ Works partially, Docker requires secrets

### 5. Release (`.github/workflows/release.yml`) ⚠️ **CONDITIONAL**
- **Purpose**: PyPI package releases
- **Triggers**: Version tags (v*)
- **Features**:
  - Package building
  - PyPI upload (requires secrets)
  - GitHub release creation
- **Status**: ⚠️ Works partially, PyPI requires secrets

### 6. Full CI (`.github/workflows/ci.disabled.yml`) ❌ **DISABLED**
- **Purpose**: Complete CI with linting, type checking, coverage
- **Status**: ❌ Disabled due to complexity and potential issues
- **To enable**: Rename to `ci.yml`

## Current Status

### ✅ Working Workflows
- **Simple CI**: Basic functionality testing
- **Test Suite**: Comprehensive testing
- **Build**: Package and Docker building

### ⚠️ Conditional Workflows
- **Deploy**: Works without secrets, full functionality with secrets
- **Release**: Works without secrets, full functionality with secrets

### ❌ Disabled Workflows
- **Full CI**: Disabled due to complexity

## Enabling Full Features

### For Docker Hub Deployment
1. Add repository secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password/token
2. The deploy workflow will automatically push images

### For PyPI Releases
1. Add repository secret:
   - `PYPI_API_TOKEN`: Your PyPI API token
2. Create version tags (e.g., `v1.0.0`) to trigger releases

### For Full CI
1. Rename `ci.disabled.yml` to `ci.yml`
2. Ensure all dependencies are properly configured
3. Test locally first with `./scripts/run_tests.sh`

## Local Testing

Before enabling workflows, test locally:

```bash
# Test basic functionality
./scripts/run_tests.sh

# Test imports
python -c "from alphascrabble import Board; print('OK')"

# Test building
python -m build

# Test Docker
docker build -t alphascrabble:test .
```

## Troubleshooting

### Python Version Issues
- Ensure Python versions are available on GitHub Actions
- Use specific patch versions if needed (e.g., `3.11.5`)

### Dependency Issues
- Check `requirements.txt` for version conflicts
- Test installation locally first

### Docker Issues
- Ensure Dockerfile is valid
- Test Docker build locally
- Check for missing dependencies

### Secret Issues
- Verify secrets are set correctly
- Check secret names match workflow expectations
- Ensure secrets have proper permissions

## Workflow Priority

1. **Start with Simple CI** - Always works
2. **Add Test Suite** - Comprehensive testing
3. **Add Build** - Package creation
4. **Add Deploy** - With secrets for full functionality
5. **Add Release** - With secrets for PyPI
6. **Enable Full CI** - When everything else works

## Recommendations

1. **Keep Simple CI active** - Provides basic validation
2. **Use Test Suite for development** - Comprehensive testing
3. **Enable Build for releases** - Package creation
4. **Add secrets gradually** - Start with Docker, then PyPI
5. **Test locally first** - Always verify before pushing
