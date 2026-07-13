#!/bin/bash
# Validation helpers for sourced Slurm runners. No commands run on source.

fail_arg() {
    echo "FATAL: $*" >&2
    return 1
}

validate_bool() {
    local name=$1 value=$2
    [[ "$value" = 0 || "$value" = 1 ]] || fail_arg "$name must be 0 or 1"
}

validate_positive_int() {
    local name=$1 value=$2
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail_arg "$name must be a positive integer"
}

validate_nonnegative_int() {
    local name=$1 value=$2
    [[ "$value" =~ ^[0-9]+$ ]] || fail_arg "$name must be a non-negative integer"
}

validate_seed() {
    local name=$1 value=$2
    [[ "$value" =~ ^-?[0-9]+$ ]] || fail_arg "$name must be an integer"
}

validate_id() {
    local name=$1 value=$2
    [[ "$value" =~ ^[a-z0-9][a-z0-9._-]{2,79}$ ]] \
        || fail_arg "$name must match [a-z0-9][a-z0-9._-]{2,79}"
}

require_single_csv_value() {
    local name=$1 value=$2
    [[ "$value" != *,* ]] || fail_arg "$name must contain exactly one value"
}

validate_csv_members() {
    local name=$1 value=$2 allowed=$3 item seen=,
    local -a items
    IFS=',' read -ra items <<< "$value"
    if [ "${#items[@]}" -eq 0 ]; then
        fail_arg "$name must not be empty"
        return 1
    fi
    for item in "${items[@]}"; do
        if [[ ",$allowed," != *",$item,"* ]]; then
            fail_arg "$name contains unsupported value: $item"
            return 1
        fi
        if [[ "$seen" = *",$item,"* ]]; then
            fail_arg "$name contains duplicate value: $item"
            return 1
        fi
        seen+="$item,"
    done
}
