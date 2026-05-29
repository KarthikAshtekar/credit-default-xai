# Maintainer Guide

## Repository Protection (Recommended)
Apply these branch protection settings on `main`:
- Require pull request before merge
- Require approvals: 1+
- Require status checks: `CI / lint-and-smoke`
- Require branches up to date before merging
- Restrict force pushes and deletion

## Release Process
- Tag stable snapshots as `vX.Y.Z`
- Update README result summaries when major pipeline changes land
