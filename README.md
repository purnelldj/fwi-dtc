# Fire Weather Index Forecaster

dev notes

## Installation

```bash
micromamba create -n fwi python=3.11
micromamba activate fwi
pip install -r models/fwi_calculator/requirements.txt
```

## Auth

```bash
touch .secrets
```
then put in .secrets file
```
export DESPAUTH_USER="your-username"
export DESPAUTH_PASSWORD="your-password"
```
add as environment variables
```bash
source .secrets
```
