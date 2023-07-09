#!/bin/bash

# FSL Setup
FSLDIR=/root/fsl
PATH=${FSLDIR}/share/fsl/bin:${PATH}
export FSLDIR PATH
. ${FSLDIR}/etc/fslconf/fsl.sh

# launch cvrmap with arguments
/opt/cvrmap/cvrmap/cvrmap "$@"
