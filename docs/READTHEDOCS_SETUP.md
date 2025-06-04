# Read the Docs Setup Guide

This document explains how to set up documentation hosting on Read the Docs for the DSS Pollution Extraction project.

## Prerequisites

- GitHub repository with your project
- Read the Docs account (free at https://readthedocs.org/)
- Admin access to your GitHub repository

## Setup Steps

### 1. Create Read the Docs Account

1. Go to https://readthedocs.org/
2. Sign up using your GitHub account
3. Grant Read the Docs access to your repositories

### 2. Import Your Project

1. Click "Import a Project" on your Read the Docs dashboard
2. Select your `dss-pollution-extraction` repository
3. Click "Next" to proceed with the import

### 3. Configure Project Settings

In your Read the Docs project settings:

#### Basic Settings
- **Name**: `dss-pollution-extraction`
- **Repository URL**: `https://github.com/MuhammadShafeeque/dss-pollution-extraction`
- **Default branch**: `main`
- **Default version**: `latest`

#### Advanced Settings
- **Documentation type**: `Sphinx Html`
- **Python interpreter**: `CPython 3.11`
- **Requirements file**: `docs/requirements.txt`
- **Install project**: `Yes` (checked)

### 4. Environment Variables (if needed)

If your documentation build requires specific environment variables:

1. Go to your project's "Environment Variables" section
2. Add any required variables

### 5. Build Configuration

The project includes a `.readthedocs.yaml` configuration file that specifies:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/source/conf.py
python:
  install:
    - requirements: docs/requirements.txt
    - requirements: requirements.txt
    - method: pip
      path: .
```

### 6. Webhook Configuration

Read the Docs should automatically configure webhooks. If not:

1. Go to your GitHub repository settings
2. Navigate to "Webhooks"
3. Add a webhook pointing to your Read the Docs project

### 7. Domain Configuration

#### Default Domain
Your documentation will be available at:
`https://dss-pollution-extraction.readthedocs.io/`

#### Custom Domain (Optional)
If you have a custom domain:

1. Go to your Read the Docs project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Update your DNS settings accordingly

## Troubleshooting

### Common Build Issues

#### Missing Dependencies
If the build fails due to missing dependencies:

1. Check `docs/requirements.txt` includes all necessary packages
2. Verify the package versions are compatible
3. Check the build logs for specific error messages

#### Import Errors
If autodoc fails to import modules:

1. Ensure the project is installed during build (`Install project: Yes`)
2. Check that mock imports are configured in `docs/source/conf.py`
3. Verify Python path configuration

#### Build Timeout
If builds are timing out:

1. Reduce the number of autodoc modules
2. Use mock imports for heavy dependencies
3. Consider building documentation locally first

### Build Logs

Access build logs to diagnose issues:

1. Go to your Read the Docs project
2. Click on "Builds"
3. Select a specific build to view logs

### Local Testing

Test documentation builds locally before pushing:

```bash
cd docs
pip install -r requirements.txt
make html
```

## Automated Updates

### GitHub Integration

Read the Docs will automatically rebuild documentation when:

1. You push to the `main` branch
2. You create a new tag/release
3. Pull requests are opened (if configured)

### Version Management

Read the Docs can build multiple versions:

1. **latest**: Latest commit on main branch
2. **stable**: Latest tagged release
3. **Tagged versions**: Specific release versions

Configure this in your project's "Versions" settings.

## Documentation URLs

Once set up, your documentation will be available at:

- **Latest**: https://dss-pollution-extraction.readthedocs.io/en/latest/
- **Stable**: https://dss-pollution-extraction.readthedocs.io/en/stable/
- **Specific version**: https://dss-pollution-extraction.readthedocs.io/en/v1.0.1/

## Additional Features

### Download Formats

Read the Docs can generate documentation in multiple formats:
- HTML (default)
- PDF
- ePub

Enable these in your project settings under "Advanced Settings".

### Search

Read the Docs provides full-text search across your documentation automatically.

### Analytics

Enable analytics to track documentation usage:

1. Go to project settings
2. Navigate to "Analytics"
3. Configure your preferred analytics service

## Maintenance

### Regular Updates

- Keep dependencies in `docs/requirements.txt` updated
- Monitor build status and fix any failures promptly
- Update documentation content regularly

### Performance Optimization

- Use mock imports for heavy scientific packages
- Optimize image sizes in documentation
- Consider caching strategies for large datasets

## Support

If you encounter issues:

1. Check Read the Docs documentation: https://docs.readthedocs.io/
2. Search Read the Docs community forum
3. File issues in the project repository
4. Contact Read the Docs support for hosting-specific issues
