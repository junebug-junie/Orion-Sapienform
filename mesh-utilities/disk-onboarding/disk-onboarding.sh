#!/usr/bin/env bash
#
# disk-onboarding.sh v2
# Full disk discovery, optional formatting, labeling, and fstab configuration
# Now supports raw devices like nvme0n1 and scratch disk optimization.

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔎 Running in DRY RUN mode (no writes will be performed)"
fi

echo "🔎 Scanning for disks and partitions..."
echo "--------------------------------------"
lsblk -o NAME,SIZE,FSTYPE,UUID,LABEL,MOUNTPOINT

# Detect ext4, vfat, and also unformatted devices (no FSTYPE)
for dev in $(lsblk -lnpo NAME,FSTYPE | awk '$2=="" || $2=="ext4" || $2=="vfat" {print $1}'); do
    FSTYPE=$(blkid -s TYPE -o value "$dev")
    UUID=$(blkid -s UUID -o value "$dev")
    LABEL=$(lsblk -no LABEL "$dev")
    CURRENT_MP=$(lsblk -no MOUNTPOINT "$dev")

    # Skip root device
    if findmnt -n -o SOURCE / | grep -q "$dev"; then
        echo "⏭ Skipping root device: $dev"
        continue
    fi

    echo
    echo "📦 Found: $dev [${FSTYPE:-UNFORMATTED}, UUID=${UUID:-<none>}, LABEL=${LABEL:-<none>}]"

    # Format if empty
    if [[ -z "$FSTYPE" ]]; then
        read -p "⚠️  $dev has no filesystem. Format as ext4 now? (y/N): " CONFIRM_FMT
        if [[ "$CONFIRM_FMT" =~ ^[Yy]$ ]]; then
            if $DRY_RUN; then
                echo "🟡 DRY RUN: would run mkfs.ext4 -F -E discard $dev"
            else
                sudo mkfs.ext4 -F -E discard "$dev"
            fi
            FSTYPE="ext4"
        else
            echo "⏭ Skipping $dev (unformatted)."
            continue
        fi
    fi

    # Prompt for label if missing or you want to change it
    read -p "Enter label for $dev [${LABEL:-none}]: " NEWLABEL
    if [[ -n "$NEWLABEL" ]]; then
        LABEL="$NEWLABEL"
        if $DRY_RUN; then
            echo "🟡 DRY RUN: would set label $LABEL on $dev"
        else
            if [[ "$FSTYPE" == "ext4" ]]; then
                sudo e2label "$dev" "$LABEL"
            elif [[ "$FSTYPE" == "vfat" ]]; then
                sudo fatlabel "$dev" "$LABEL" || sudo dosfslabel "$dev" "$LABEL"
            fi
        fi
    fi

    # Build mountpoint suggestion
    MOUNTPOINT=${CURRENT_MP:-/mnt/${LABEL,,}}
    MOUNTOPTS="defaults,nofail,noatime,discard"

    echo "🔧 Suggested fstab entry:"
    echo "LABEL=$LABEL  $MOUNTPOINT  $FSTYPE  $MOUNTOPTS  0 2"

    read -p "Add to /etc/fstab and mount now? (y/N): " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        if $DRY_RUN; then
            echo "🟡 DRY RUN: would create $MOUNTPOINT, append to /etc/fstab, and mount"
        else
            sudo mkdir -p "$MOUNTPOINT"
            if ! grep -q "LABEL=$LABEL" /etc/fstab; then
                echo "LABEL=$LABEL  $MOUNTPOINT  $FSTYPE  $MOUNTOPTS  0 2" | sudo tee -a /etc/fstab
            fi
            sudo mount -a
        fi
    fi
done

echo
echo "✅ Done. Test with: findmnt | grep /mnt/"

