#!/bin/bash
set -e

SNPE_TARGET_ROOT_DIR=/data/local/tmp/snpeexample

function source_env_target() {
    export SNPE_TARGET_ARCH=aarch64-oe-linux-gcc11.2
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SNPE_TARGET_ROOT_DIR/$SNPE_TARGET_ARCH/lib
    export PATH=$PATH:$SNPE_TARGET_ROOT_DIR/$SNPE_TARGET_ARCH/bin
    export ADSP_LIBRARY_PATH="$SNPE_TARGET_ROOT_DIR/dsp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
}

source_env_target
