# Disk Onboarding Utility

Automates discovery, labeling, and fstab configuration for new drives.

## Usage

```bash
./disk-onboarding.sh
```

- Detects ext4/vfat partitions
- Prompts for labels (e.g. `DOCKER`, `STORAGE`)
- Suggests `/etc/fstab` entries
- Confirms before writing
- Creates mountpoints if missing
